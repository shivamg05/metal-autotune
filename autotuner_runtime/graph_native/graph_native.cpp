// Functional graph replacement: array values and the input graph are never mutated.
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <functional>
#include <cstring>
#include <sstream>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include "mlx/array.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"

namespace nb = nanobind;
namespace mx = mlx::core;
using Bindings = std::unordered_map<uintptr_t, mx::array>;
using Ids = std::unordered_set<uintptr_t>;

bool equivalent(const mx::array& a, const mx::array& b) {
  if (typeid(a.primitive()) != typeid(b.primitive()) ||
      a.primitive().stream() != b.primitive().stream() ||
      a.sibling_position() != b.sibling_position() ||
      a.siblings().size() != b.siblings().size() ||
      a.inputs().size() != b.inputs().size()) return false;
  if (a.primitive_ptr() == b.primitive_ptr()) return true;
  // CustomKernel does not implement Primitive::is_equivalent in MLX 0.32.2.
  // Its complete public state includes source, launch, layout and compile options.
  // MLX does not export CustomKernel RTTI, so identify its public primitive name
  // after the dynamic type equality check above.
  if (std::string(a.primitive().name()) == "CustomKernel")
    return static_cast<mx::fast::CustomKernel&>(a.primitive()).state() ==
        static_cast<mx::fast::CustomKernel&>(b.primitive()).state();
  return a.primitive().is_equivalent(b.primitive());
}

bool match(const mx::array& pattern, const mx::array& value,
           const Ids& parameters, Bindings& bound) {
  std::vector<std::pair<mx::array, mx::array>> pending{{pattern, value}};
  while (!pending.empty()) {
    auto [p, a] = pending.back();
    pending.pop_back();
    if (p.shape() != a.shape() || p.dtype() != a.dtype()) return false;
    if (auto it = bound.find(p.id()); it != bound.end()) {
      if (it->second.id() != a.id()) return false;
      continue;
    }
    if (!parameters.contains(p.id())) {
      if (!p.has_primitive()) {
        // Python scalar literals become separate materialized scalar arrays.
        // Compare their bytes without evaluation, synchronization or dtype casts.
        if (p.id() != a.id() &&
            (p.size() != 1 || a.has_primitive() || !p.is_available() || !a.is_available() ||
             std::memcmp(p.data<char>(), a.data<char>(), p.itemsize()) != 0)) return false;
      } else {
        if (!a.has_primitive() || a.status() != mx::array::unscheduled ||
            p.status() != mx::array::unscheduled || !equivalent(p, a)) return false;
        // Preserve operation identity across parallel branches. Boundary
        // parameters remain free to alias (for example, x + x).
        for (const auto& [id, value] : bound)
          if (!parameters.contains(id) && id != p.id() && value.id() == a.id()) return false;
        for (size_t i = 0; i < p.inputs().size(); ++i)
          pending.emplace_back(p.inputs()[i], a.inputs()[i]);
        auto po = p.outputs(), ao = a.outputs();
        for (size_t i = 0; i < po.size(); ++i) {
          if (po[i].shape() != ao[i].shape() || po[i].dtype() != ao[i].dtype()) return false;
          // Bind siblings together: outputs of separate calls cannot impersonate
          // outputs of one multi-output primitive.
          auto it = bound.find(po[i].id());
          if (it != bound.end() && it->second.id() != ao[i].id()) return false;
          for (const auto& [id, value] : bound)
            if (!parameters.contains(id) && id != po[i].id() && value.id() == ao[i].id()) return false;
          if (po[i].id() != p.id()) bound.emplace(po[i].id(), ao[i]);
        }
      }
    }
    bound.emplace(p.id(), a);
  }
  return true;
}

std::vector<mx::array> graph_nodes(const std::vector<mx::array>& roots) {
  std::vector<mx::array> nodes, pending = roots;
  Ids seen;
  while (!pending.empty()) {
    auto a = pending.back(); pending.pop_back();
    if (!seen.insert(a.id()).second) continue;
    nodes.push_back(a);
    if (a.has_primitive() && a.status() == mx::array::unscheduled)
      for (const auto& input : a.inputs()) pending.push_back(input);
  }
  return nodes;
}

std::string node_key(const mx::array& a) {
  std::ostringstream out;
  out << typeid(a.primitive()).name() << ':' << int(a.dtype().val()) << ':' << a.sibling_position();
  for (auto dim : a.shape()) out << ':' << dim;
  return out.str();
}

struct Occurrence {
  std::vector<mx::array> inputs, outputs;
};

std::tuple<std::vector<mx::array>, int> rewrite(
    const std::vector<mx::array>& roots,
    const std::vector<mx::array>& patterns,
    const std::vector<mx::array>& parameters,
    const nb::callable& replacement,
    const std::vector<std::optional<mx::array>>& anchors) {
  if (patterns.empty()) throw std::invalid_argument("graph pattern needs at least one output");
  Ids leaves;
  for (const auto& p : parameters) {
    if (!leaves.insert(p.id()).second)
      throw std::invalid_argument("graph parameters must be distinct arrays");
  }
  for (const auto& p : patterns)
    if (!p.has_primitive() || p.status() != mx::array::unscheduled || leaves.contains(p.id()))
      throw std::invalid_argument("graph pattern outputs must be unevaluated operations");

  if (!anchors.empty() && anchors.size() != parameters.size())
    throw std::invalid_argument("graph anchors must align with parameters");
  Bindings fixed;
  for (size_t i = 0; i < anchors.size(); ++i)
    if (anchors[i]) fixed.emplace(parameters[i].id(), *anchors[i]);

  auto nodes = graph_nodes(roots);
  std::unordered_map<std::string, std::vector<mx::array>> index;
  for (const auto& a : nodes)
    if (a.has_primitive() && a.status() == mx::array::unscheduled)
      index[node_key(a)].push_back(a);

  std::vector<Occurrence> occurrences;
  std::unordered_map<uintptr_t, size_t> owners;
  // Start at the last boundary output; it often depends on the earlier outputs.
  const auto& anchor = patterns.back();
  for (const auto& a : index[node_key(anchor)]) {
    if (owners.contains(a.id())) continue;
    Bindings bound = fixed;
    if (!match(anchor, a, leaves, bound)) continue;
    std::function<bool(size_t, Bindings&)> complete;
    complete = [&](size_t i, Bindings& bindings) {
      if (i == patterns.size()) return true;
      const auto& p = patterns[i];
      if (auto it = bindings.find(p.id()); it != bindings.end()) {
        if (owners.contains(it->second.id())) return false;
        return complete(i + 1, bindings);
      }
      for (const auto& candidate : index[node_key(p)]) {
        if (owners.contains(candidate.id())) continue;
        auto trial = bindings;
        if (match(p, candidate, leaves, trial) && complete(i + 1, trial)) {
          bindings = std::move(trial);
          return true;
        }
      }
      return false;
    };
    if (!complete(0, bound)) continue;
    Occurrence found;
    for (const auto& p : parameters) {
      if (!bound.contains(p.id()))
        throw std::invalid_argument("graph parameter is not used by the pattern");
      found.inputs.push_back(bound.at(p.id()));
    }
    Ids internal;
    for (const auto& [id, value] : bound)
      if (!leaves.contains(id)) internal.insert(value.id());
    bool cyclic = false;
    for (const auto& input : found.inputs) cyclic |= internal.contains(input.id());
    if (cyclic) continue;
    for (const auto& p : patterns) found.outputs.push_back(bound.at(p.id()));
    for (const auto& output : found.outputs) owners.emplace(output.id(), occurrences.size());
    occurrences.push_back(std::move(found));
  }

  // Traverse the new dependencies, which may join several original output branches.
  // Mapping all outputs at once preserves each branch and all outside consumers.
  Bindings memo;
  Ids active;
  std::vector<bool> declined(occurrences.size(), false);
  std::vector<std::pair<mx::array, bool>> pending;
  for (const auto& root : roots) pending.emplace_back(root, false);
  int hits = 0;
  while (!pending.empty()) {
    auto [a, expanded] = pending.back(); pending.pop_back();
    if (memo.contains(a.id())) continue;
    if (!a.has_primitive() || a.status() != mx::array::unscheduled) {
      memo.emplace(a.id(), a);
      continue;
    }
    auto owner = owners.find(a.id());
    bool replacing = owner != owners.end() && !declined[owner->second];
    auto dependencies = replacing ? occurrences[owner->second].inputs : a.inputs();
    if (!expanded) {
      if (!active.insert(a.id()).second)
        throw std::invalid_argument("graph replacement would introduce a dependency cycle");
      pending.emplace_back(a, true);
      for (const auto& input : dependencies) pending.emplace_back(input, false);
      continue;
    }
    active.erase(a.id());
    std::vector<mx::array> inputs;
    bool changed = false;
    for (const auto& input : dependencies) {
      inputs.push_back(memo.at(input.id()));
      changed |= inputs.back().id() != input.id();
    }
    if (replacing) {
      const auto& occurrence = occurrences[owner->second];
      auto result = replacement(inputs);
      if (result.is_none()) {
        declined[owner->second] = true;
        pending.emplace_back(a, false);
        continue;
      }
      auto outputs = nb::cast<std::vector<mx::array>>(result);
      if (outputs.size() != occurrence.outputs.size())
        throw std::invalid_argument("graph replacement changed output count");
      for (size_t i = 0; i < outputs.size(); ++i) {
        const auto& old = occurrence.outputs[i];
        if (outputs[i].shape() != old.shape() || outputs[i].dtype() != old.dtype())
          throw std::invalid_argument("graph replacement changed output shape or dtype");
        if (auto it = memo.find(old.id()); it != memo.end() && it->second.id() != outputs[i].id())
          throw std::invalid_argument("graph replacement changed aliased outputs");
        memo.emplace(old.id(), outputs[i]);
      }
      ++hits;
      continue;
    }
    auto old_outputs = a.outputs();
    // A sibling can have its own replacement even when this output stays
    // original. Do not cache that sibling prematurely and bypass its callback.
    auto keep_output = [&](const mx::array& old, const mx::array& value) {
      auto next = owners.find(old.id());
      if (next == owners.end() || declined[next->second]) memo.emplace(old.id(), value);
    };
    if (!changed) {
      for (const auto& old : old_outputs) keep_output(old, old);
    } else {
      std::vector<mx::Shape> shapes;
      std::vector<mx::Dtype> dtypes;
      for (const auto& old : old_outputs) {
        shapes.push_back(old.shape()); dtypes.push_back(old.dtype());
      }
      auto outputs = mx::array::make_arrays(shapes, dtypes, a.primitive_ptr(), inputs);
      for (size_t i = 0; i < outputs.size(); ++i) {
        outputs[i].set_tracer(old_outputs[i].is_tracer());
        keep_output(old_outputs[i], outputs[i]);
      }
    }
  }
  std::vector<mx::array> outputs;
  for (const auto& root : roots) outputs.push_back(memo.at(root.id()));
  return {outputs, hits};
}

std::tuple<bool, std::string> same_structure(
    const std::vector<mx::array>& original, const std::vector<mx::array>& rewritten) {
  if (original.size() != rewritten.size()) return {false, "root count"};
  std::vector<std::pair<mx::array, mx::array>> pending;
  for (size_t i = 0; i < original.size(); ++i) pending.emplace_back(original[i], rewritten[i]);
  std::unordered_map<uintptr_t, uintptr_t> mapping, reverse;
  while (!pending.empty()) {
    auto [a, b] = pending.back(); pending.pop_back();
    if (auto it = mapping.find(a.id()); it != mapping.end()) {
      if (it->second != b.id()) return {false, "shared node duplicated"};
      continue;
    }
    if (auto it = reverse.find(b.id()); it != reverse.end() && it->second != a.id())
      return {false, "distinct nodes merged"};
    mapping.emplace(a.id(), b.id()); reverse.emplace(b.id(), a.id());
    if (a.shape() != b.shape() || a.dtype() != b.dtype()) return {false, "shape or dtype"};
    if (a.has_primitive() != b.has_primitive()) return {false, "primitive versus leaf"};
    if (!a.has_primitive() || a.status() != mx::array::unscheduled) {
      if (a.id() != b.id()) return {false, "different leaf"};
      continue;
    }
    if (!equivalent(a, b)) return {false, std::string("primitive mismatch: ") + a.primitive().name()};
    for (size_t i = 0; i < a.inputs().size(); ++i) pending.emplace_back(a.inputs()[i], b.inputs()[i]);
  }
  return {true, "same operations, attributes, leaves and sharing"};
}

NB_MODULE(_graph_native, m) {
  m.def("rewrite", &rewrite, nb::arg("roots"), nb::arg("patterns"),
        nb::arg("parameters"), nb::arg("replacement"),
        nb::arg("anchors") = std::vector<std::optional<mx::array>>{});
  m.def("same_structure", &same_structure);
  m.def("array_id", [](const mx::array& a) { return a.id(); });
}
