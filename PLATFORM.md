# PLATFORM.md

Verified platform facts, each proven by a spike in `spikes/` run on this machine.
Machine: Apple M4, 24 GB unified memory, macOS (Darwin 25.3.0). mlx 0.32.2 (pinned in
pyproject.toml). Python 3.12 via uv. Spike outputs archived in `spikes/out/logs/`.

Line format is the spike output itself: `FACT <slug>: PASS|FAIL|INFO - detail`.
PASS means the implementation plan's claim held; FAIL means it did not (the design
adjustment is recorded below and in IMPLEMENTATION_PLAN.md); INFO is a measured
number with no prior claim. Each spike graduates into pinned tests in
`tests/test_platform_*.py` so an mlx upgrade fails loudly.

## spike_01_patchability: Patchability (mx module functions, array dunders)

- FACT mlx-version: INFO - mlx 0.32.2 on darwin, python 3.12.13
- FACT array-nanobind-heaptype: PASS - metatype=nanobind.nb_type, HEAPTYPE=True, IMMUTABLETYPE=False
- FACT array-dunder-inventory: INFO - 51/58 probed names in vars(mx.array); missing: __rmatmul__,__rand__,__ror__,__rxor__,__rlshift__,__rrshift__,__pos__; __hash__ inherited from object (pointer hash)
- FACT module-add-not-plus: PASS - a + b with only mx.add patched: ran=True, patched mx.add called=False
- FACT array-dunder-setattr: PASS - setattr accepted on 51/51 present names (+__hash__ added)
- FACT module-setattr-nn-visible: PASS - nn.Linear(bias=True) hit patched module-level op: True (recorded: mx.addmm)
- FACT linear-op-usage: INFO - bias=True records mx.addmm; bias=False records __matmul__ (x @ W.T, not mx.matmul)
- FACT slot-propagation-core: PASS - a+b, 2.0*x, 2.0+x, a@b, a[i], a[i]=v, a==b all intercepted via C slots
- FACT wrapper-delegation-correct: PASS - (a + b) under full patch = [7.0, 7.0, 7.0, 7.0, 7.0, 7.0], expected [7.0]*6
- FACT intercept-arithmetic: PASS - intercepted 13/13: __add__,__sub__,__mul__,__truediv__,__floordiv__,__mod__,__pow__,__matmul__,__and__,__or__,__xor__,__lshift__,__rshift__
- FACT intercept-reflected: PASS - intercepted 7/13: __radd__,__rsub__,__rmul__,__rtruediv__,__rfloordiv__,__rmod__,__rpow__; no dunder and op raises unpatched (nothing to trace): __rmatmul__,__rand__,__ror__,__rxor__,__rlshift__,__rrshift__
- FACT intercept-inplace: PASS - intercepted 13/13: __iadd__,__isub__,__imul__,__itruediv__,__ifloordiv__,__imod__,__ipow__,__imatmul__,__iand__,__ior__,__ixor__,__ilshift__,__irshift__
- FACT iadd-fold: INFO - a += b hits __iadd__ (no fold)
- FACT intercept-unary: PASS - intercepted 3/4: __neg__,__abs__,__invert__; no dunder and op raises unpatched (nothing to trace): __pos__
- FACT intercept-comparison: PASS - intercepted 6/6: __eq__,__ne__,__lt__,__le__,__gt__,__ge__
- FACT intercept-indexing: PASS - intercepted 2/2: __getitem__,__setitem__
- FACT intercept-conversion: PASS - intercepted 5/5: __len__,__iter__,__bool__,__int__,__float__
- FACT intercept-plain-methods: PASS - intercepted 2/2: reshape,sum
- FACT swapped-operand-no-reflected: PASS - unchanged by patching, all raise TypeError pre and post: __rmatmul__: pre=TypeError patched=TypeError; __rand__: pre=TypeError patched=TypeError; __ror__: pre=TypeError patched=TypeError; __rxor__: pre=TypeError patched=TypeError; __rlshift__: pre=TypeError patched=TypeError; __rrshift__: pre=TypeError patched=TypeError
- FACT eq-hash-dict-set: PASS - dict lookups=True, set membership=True, patched __hash__ called 9x (identity keys; == is elementwise so only same-object keys are meaningful, patched or not)
- FACT uninstall-restores-identity: PASS - identity mismatches: none; wrappers silent after restore=True; outputs still correct=True
- FACT uninstall-restores-behavior: PASS - swapped-operand ops behave exactly as before patching

**Deviations from the plan:** No plan claim failed; all PASS on mlx 0.32.2. Three new facts worth recording in PLATFORM.md: (1) mx.array lacks __rmatmul__, __rand__, __ror__, __rxor__, __rlshift__, __rrshift__, and __pos__, but each corresponding op (list @ a, 2 & a, +a, etc.) raises TypeError unpatched too, so these are not tracer holes and patching does not change their behavior; the plan's 'reflected need their own patches' applies to the 7 arithmetic reflected dunders, which all exist and intercept. (2) a += b hits __iadd__ and never folds to __add__, so the tracer must patch all 13 in-place dunders (all exist). (3) nn.Linear(bias=False) uses x @ W.T via array.__matmul__, never mx.matmul, so module-level patching alone misses bias-free Linear; __hash__ is not in vars(mx.array) (inherited object pointer hash) and a delegating __hash__ added via setattr uninstalls cleanly with delattr.

## spike_02_module_wrapping: Module wrapping (per-subclass __call__)

- FACT module-base-no-own-call: PASS - '__call__' in vars(nn.Module) is False; MRO ['Module', 'dict', 'object']
- FACT base-class-patch-does-not-fire: PASS - patched nn.Module.__call__, called nn.Linear: wrapper fired 0 times, output bitwise equal True
- FACT base-class-unpatch-clean: PASS - after del, '__call__' in vars(nn.Module) is False
- FACT subclass-patch-intercepts-linear: PASS - wrapper fired 1x for Linear, output bitwise equal True
- FACT subclass-patch-intercepts-layernorm: PASS - wrapper fired 1x for LayerNorm, output bitwise equal True
- FACT self-identity-dispatch: PASS - recorded ids [4366604544, 4366835920] match [id(lin), id(lin2)]=True; instance outputs differ True
- FACT unpatch-restores-exactly: PASS - class dict entries identical to originals True; wrapper fired 0x after restore; output bitwise equal True
- FACT instance-assign-does-not-intercept: PASS - assigned lin.__call__ on the instance: fired 0x, call still dispatched via type, output bitwise equal True
- FACT instance-assign-attr-location: INFO - the assigned attribute landed in the instance __dict__; dunder lookup bypasses it
- FACT wrap-top-level-module-model: PASS - wrapper fired 1x, output bitwise equal True (Sequential callable)
- FACT wrap-top-level-function-model: PASS - wrapper fired 1x, output bitwise equal True (function callable)

**Deviations from the plan:** none. All plan 5.1 claims held on mlx 0.32.2. One detail worth recording in PLATFORM.md: per-instance __call__ assignment on an mlx Module is silently accepted (Module.__setattr__ routes it into the instance __dict__, not the dict storage), so a tracer bug that patched instances instead of classes would fail silently with zero interception, exactly as the plan warns.

## spike_03_weakref_retention: Weakref, GC, and retention detection

- FACT weakref-accepted: PASS - weakref.ref succeeded on lazy (a+b) and evaluated (mx.eval'd) arrays, type=array
- FACT weakref-lazy-dies: PASS - lazy array: alive while referenced=True, dead after del+gc.collect()=True
- FACT weakref-evaluated-dies: PASS - evaluated array: alive while referenced=True, dead after del+gc.collect()=True
- FACT module-state-attr-reserved: INFO - assigning self.state on a Module subclass raises AttributeError (property 'state' of 'run.<locals>.StateName' object has no deleter); toy fixtures must avoid that attribute name
- FACT walk-array-count: INFO - snapshot walk reached 5 distinct arrays on the model object
- FACT retained-and-consumed: PASS - k appended to cache AND used in later op; walk found it at model.cache[0]; weakref alive after forward=True; out=f(k) evaluated to shape (1, 4)
- FACT walk-finds-nested-containers: PASS - list=model.cache[0], dict-nested=model.store.hist[0], tuple=model.snap[0], parameter=model.proj.weight
- FACT walk-negative-control: PASS - step output (not stored on model) found by walk=False
- FACT retention-clears-on-drop: PASS - after clearing cache/store/snap and gc.collect(), k weakref dead=True

**Deviations from the plan:** None against the plan's claims: all plan 5.1 liveness claims held on mlx 0.32.2. One incidental platform fact found while building the spike and pinned as an INFO line: mlx nn.Module reserves `state` as a property with no deleter, so assigning `self.state` on any Module subclass raises AttributeError; toy fixtures (and the fixture zoo) must avoid that attribute name.

## spike_04_delivery: Delivery micro-spike (the bind mechanism in miniature)

- FACT module-state-name-reserved: INFO - nn.Module.state is a property on 0.32.2: True; assigning self.state in a subclass raises AttributeError, so the toy's state attribute is named acc
- FACT record-op-stream: PASS - one Block call recorded as ['multiply', 'add', 'exp', 'sin', 'matmul', 'sum', 'add', 'add'], matching the hand-derived sequence
- FACT parent-setattr-swap: PASS - parent setattr installed a plain-object wrapper (stored in __dict__=True, module dict entry popped=True); lookup returns the wrapper
- FACT identity-wrapper-bitwise: PASS - 3 calls routed through the wrapper; every output mx.array_equal to baseline
- FACT identity-wrapper-state: PASS - state array after each of 3 calls mx.array_equal to baseline (state updates flowed through the replayed ops)
- FACT wrapper-getattr-delegation: PASS - wrapper.a, wrapper.b.weight, wrapper.acc resolve to the wrapped original's own objects
- FACT identity-retrace-matches: PASS - retrace of the identity wrapper recorded ['multiply', 'add', 'exp', 'sin', 'matmul', 'sum', 'add', 'add'], identical to baseline
- FACT metal-kernel-signature: PASS - constructed from body-only source with compile_options math_mode=safe; call is keyword-only with inputs/output_shapes/output_dtypes/grid/threadgroup, grid in total threads
- FACT transcendental-precision: INFO - max abs diff vs library sin(exp(x)): {'metal/safe': 4.172325134277344e-07, 'metal/fast': 4.172325134277344e-07, 'precise/safe': 0.0, 'precise/fast': 0.0}; precision comes from the metal:: vs metal::precise:: namespace, not from math_mode
- FACT compile-error-at-eval: PASS - construction and call both accepted a broken kernel (hyphenated name); RuntimeError surfaced only at mx.eval of the probe output: [metal::Device] Unable to build metal library from source
- FACT fused-kernel-matches: PASS - fused sin(exp(x)) kernel over 3 stateful calls within fp32 rtol=1e-05 atol=1e-06; max abs diff 0
- FACT fused-kernel-bitwise: INFO - fused kernel bitwise-equal to library exp-then-sin: True (max abs diff 0); order-preserving math did reproduce library bits
- FACT retrace-cut-ops-gone: PASS - fused retrace recorded ['multiply', 'add', 'custom_kernel', 'matmul', 'sum', 'add', 'add']: exp and sin gone, one custom kernel call in their place, neighbors intact and in order
- FACT rollback-restores: PASS - parent setattr restored the original module; trajectory bitwise equal to baseline and retrace stream identical

**Deviations from the plan:** Four findings beyond the plan's claims, none breaking the design. (1) mlx 0.32.2 nn.Module reserves `state` as a property with no deleter, so a Module subclass cannot assign self.state; toy fixtures and real stateful models must use another name (the tracer will never see a Module named `state` attribute, since mlx itself rejects it). (2) Plain metal:: transcendentals (exp, sin) differ from MLX's library kernels by ~4e-7 per element and math_mode (safe vs fast) does NOT change that; metal::precise:: variants reproduce library bits exactly. Plan 5.10 treats math_mode as the precision knob; for transcendentals the actual lever is the metal:: vs metal::precise:: namespace. The judge prompt and naive lowering should use precise:: where library-bit fidelity matters, and pinning math_mode=safe alone does not buy it. (3) The kernel name must be a valid C identifier: it is pasted into the generated signature, a hyphenated name breaks compilation, and both construction and call accept it silently with the RuntimeError surfacing only at probe eval (this confirms plan 5.10's error-surfacing claim; the name constraint itself is new and belongs in static checks). (4) Swapping a plain-object (non-Module) wrapper in by parent setattr stores it in the parent's instance __dict__ and Module.__setattr__ pops the old module-dict entry, so swap and rollback are clean in both directions, but the wrapped child disappears from root.parameters()/module-tree walks while installed; anything that walks the tree post-install (e2e checks, a second capture pass) must account for that or the generated wrappers should subclass nn.Module. The core plan 5.2 claims all held: identity replay is bitwise invisible including state across repeated calls, the retrace literally loses the cut ops and gains one custom-kernel node, and rollback restores baseline exactly.

## spike_05_module_swap: Module swap semantics

- FACT mlx-version: INFO - mlx 0.32.2, device Device(gpu, 0)
- FACT parent-setattr-named-child: PASS - output 2.0->3.0 (want 2.0->3.0), new in named_modules=True, old removed=True
- FACT update-modules-rejects-numeric-dict-keys: PASS - key '3': ValueError(Module does not have sub-module named "3".); key 3: ValueError(Module does not have sub-module named "3".); output before/after attempts 24.0/24.0
- FACT update-modules-numeric-keys-nonstrict: INFO - strict=False silently no-ops on dict key '3': output stays 24.0
- FACT update-modules-list-form: PASS - spec {'layers': [{}, {}, {}, new]} gives output 60.0 (want 60.0), layers[3] is new=True
- FACT list-index-assignment: PASS - model.layers[3] = new gives output 60.0 (want 60.0), new in named_modules=True, old removed=True
- FACT wrapper-getattr-delegation: PASS - output bitwise equal after swap=True (11.0 vs 11.0), child.weight=2.0 (want 2.0), child.sub.attr=7.0 (want 7.0)
- FACT wrapped-tree-report: INFO - after wrapper swap named_modules=['child', 'child.wrapped', 'child.wrapped.sub'], parameters=['child.wrapped.sub.attr', 'child.wrapped.weight']
- FACT plain-object-swap: INFO - non-Module wrapper via setattr: model output 2.0 through wrapper (calls=1), but named_modules=[] and parameters=[] (dropped from module tree)

**Deviations from the plan:** None. Every plan 5.3 / M0 claim held on mlx 0.32.2: parent setattr swaps a named child (new instance in named_modules, old gone, model uses it); update_modules with strict=True rejects numeric path segments as dict keys (both '3' and 3 raise ValueError, tree untouched) and accepts the list-form spec ({'layers': [{}, {}, {}, new]} with {} placeholders); plain list-index assignment (model.layers[3] = new) also works and the tree reports it; an nn.Module wrapper delegating __getattr__ to the wrapped module survives parent code reaching through one and two levels (self.child.weight, self.child.sub.attr) with bitwise-identical output. Two hazards measured beyond the claims: (1) update_modules with strict=False silently no-ops on numeric dict keys, so install code must never rely on non-strict mode to surface a bad address; (2) assigning a wrapper that is NOT an nn.Module subclass makes Module.__setattr__ pop the child from the module dict, so the model calls the wrapper but the whole subtree disappears from parameters() and named_modules() - generated bind wrappers must subclass nn.Module. Also, Module.__getattr__ on the wrapper is only reached after normal lookup fails, and a delegating override must raise AttributeError (not KeyError) before 'wrapped' is set or Module.__setattr__'s hasattr probe breaks during __init__.

## spike_06_metal_kernel: metal_kernel behavior

- FACT mlx_version: INFO - mlx 0.32.2 on darwin, default device Device(gpu, 0)
- FACT construction_signature: PASS - doc signature matches plan 5.10: metal_kernel(name: str, input_names: collections.abc.Sequence[str], output_names: collections.abc.Sequence[str], source: str, header: str = '', ensure_row_contiguous: bool = True, atomic_outputs: bool = False, compile_options: object | None = None) -> object
- FACT call_keyword_only: PASS - positional call raises TypeError; keyword call with inputs/output_shapes/output_dtypes/grid/threadgroup/template/init_value/verbose runs correctly
- FACT grid_total_threads: PASS - grid=(50,1,1) tg=(32,1,1): thread ids cover 0..49 and threads_per_grid.x=50 (dispatchThreads semantics, grid is TOTAL THREADS)
- FACT compile_error_at_eval: PASS - RuntimeError at mx.eval only; message reports line 475 for an error on body line 3
- FACT error_line_offset: INFO - offset=472 lines (reported minus body line) for 1-in/1-out kernel, constant across error positions=True; with 2 inputs offset=473 (offset depends on generated signature length, so the harness must compute it per kernel)
- FACT math_mode_options: PASS - compile_options math_mode accepts and runs: safe=ok, relaxed=ok, fast=ok
- FACT init_value_nan_poison: PASS - init_value=nan: unwritten half reads back all-NaN and written half is exact, 5/5 runs with the pool pre-dirtied
- FACT shape_buffer_injection: PASS - referencing inp_shape/inp_strides/inp_ndim auto-injects buffers (present in generated signature) and values are correct: ndim/shape/strides=[2.0, 3.0, 4.0, 4.0, 1.0]
- FACT grid_builtins_conditional: PASS - generated signature includes thread_position_in_grid only when the source mentions it; a bare kernel's signature has no grid built-ins and no shape buffers
- FACT row_contiguous_true_copy: PASS - ensure_row_contiguous=True on a transposed view: kernel sees a contiguous row-major copy, flat read equals the transposed logical order [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
- FACT row_contiguous_false_wrong: PASS - ensure_row_contiguous=False with naive flat indexing is silently wrong on the transposed view (kernel read the raw untransposed buffer: True); no error was raised
- FACT same_name_one_process: PASS - two kernels with the same name and different source behave independently in one process, interleaved results [1.0, 2.0, 1.0, 2.0]
- FACT cross_process_cache: PASS - 4 sequential processes, same kernel name, source differs only in a constant: each read back its own value ['111.0', '222.0', '111.0', '333.0']; on-disk shader cache served no stale binary
- FACT float_atomics_nondeterminism: PASS - 1M-element atomic float sum on fixed input: 10/10 distinct bit patterns, spread 0.00222778 (nondeterministic across runs, as the plan expects)

**Deviations from the plan:** Two findings refine plan 5.10 rather than contradict it. (1) Error line mapping: the compiler reports errors against file 'mlx/backend/metal/kernels/utils.h' with a line number counting the full concatenated program (utils.h + generated signature + body). The measured offset is 472 lines for a minimal 1-input/1-output kernel and is constant across error positions in one kernel, but it grows with the generated signature (473 with 2 inputs, and it would grow further with extra outputs, mentioned grid built-ins, or injected shape buffers). So the harness cannot subtract one fixed auto-header constant; it must compute the offset per kernel from that kernel's generated signature, or probe it with a deliberate error. (2) Cross-process shader cache: the previously untested hazard is clear on 0.32.2 with default caching. Four sequential fresh processes using the same kernel name with sources differing only in one constant each got their own binary; no stale result was ever served. Also note float atomics showed 10/10 distinct bit patterns on a 1M-element sum, confirming the nondeterminism the determinism gate exists for.

## spike_07_oob_validation: OOB access and validation env vars

- FACT oob_read_no_validation: PASS - in-bounds control all-ones=True; near(+48KB past 16KB buf): no fault, zeros=4096/4096 ones=0 nans=0 sample=[0.0, 0.0, 0.0, 0.0]; far(+256MB): no fault, zeros=4096/4096 ones=0 nans=0 sample=[0.0, 0.0, 0.0, 0.0]; child rc=0
- FACT oob_read_with_validation: FAIL - stderr signals=none (stderr: only the Metal GPU Validation Enabled banner); near: no fault, zeros=4096/4096 ones=0 nans=0 sample=[0.0, 0.0, 0.0, 0.0]; identical to unvalidated run; child rc=0
- FACT oob_read_validation_stderr_report: INFO - +MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 at launch: detectable=True, stderr signals=['invalid device load'], first line: Invalid device load at offset 69632, executing kernel function: "custom_kernel_spike07_read_near_float_float"; near: no fault, zeros=4096/4096 ones=0 nans=0 sample=[0.0, 0.0, 0.0, 0.0]
- FACT shader_validation_launch_time_only: PASS - both validation vars set mid-process (readback='1'), fresh pipeline compiled after: post-set OOB run no fault, zeros=4096/4096 ones=0 nans=0 sample=[0.0, 0.0, 0.0, 0.0]; signals=none (same config at launch showed ['invalid device load']); child rc=0
- FACT capture_launch_time_only: PASS - no env at launch: start_capture before set ok=False (RuntimeError('[metal::start_capture] Failed to start: Capture layer is not inser), after os.environ set ok=False (RuntimeError('[metal::start_capture] Failed to start: Capture layer is not inser); env at launch: capture ok=True trace file created=True; rcs=0,0
- FACT oob_write_no_validation: PASS - no fault; own output intact=True; corrupted neighbor buffers=0/128 sentinel hits=0 bad value sample=[] (64KB written past a 16KB output)
- FACT oob_write_with_validation: INFO - validation+stderr-report at launch: stderr signals=['invalid device store'], first line: Invalid device store at offset 42112, executing kernel function: "custom_kernel_spike07_write_floatc_float"; own output intact=True, corrupted neighbors=0; child rc=0

**Deviations from the plan:** Plan section 8 defines validate mode as MTL_SHADER_VALIDATION=1 alone; measured on mlx 0.32.2 / macOS, that alone gives NO harness-detectable signal for an OOB access: no stderr diagnostic (only the 'Metal GPU Validation Enabled' banner), no exception, exit 0, and zerofilled values identical to the unvalidated run; nothing appeared in the unified log via `log show` either. The harness's validate mode must ALSO set MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 at launch and parse child stderr for 'Invalid device load'/'Invalid device store' lines, which carry kernel name, byte offset, buffer length, and a source location. `man MetalValidation` corroborates: reporting defaults to os_log, and MTL_SHADER_VALIDATION_FAIL_MODE defaults to 'zerofill' (invalid reads return 0, invalid writes dropped), so validation never faults the process. Also worth noting: an OOB write without validation was silently absorbed, with zero corruption observed across 128 probed 16KB neighbor buffers, so OOB writes are undetectable in score mode; correctness must rest entirely on value comparison plus validate-mode stderr parsing, exactly as the plan's 'correctness is value comparison only' law assumes. All launch-time-only claims (shader validation and capture) held, including with a fresh pipeline compiled after the mid-process environment change. Minor nondeterminism in output detail only: the byte offset in Metal's first reported invalid-store line varies run to run (whichever thread faults first); FACT statuses are stable.

## spike_08_include_flattener: Shipped MSL and the include flattener

- FACT mlx-version: INFO - mlx 0.32.2 at /Users/shivamgarg/dev/metal-autotune/.venv/lib/python3.12/site-packages/mlx
- FACT shipped-msl-tree: PASS - 90 .h files, 924458 bytes under /Users/shivamgarg/dev/metal-autotune/.venv/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels; steel GEMM, sdpa, quantized, reduction all present
- FACT msl-tree-size: INFO - kernels MSL subtree 924458 bytes; whole include tree 3346220 bytes (plan 5.9 says ~4MB under kernels/, which matches the whole tree, not kernels/); steel/gemm/gemm.h=8475B, steel/attn/attn.h=8532B, sdpa_vector.h=17572B, quantized.h=86302B, reduction/reduce_row.h=11321B, utils.h=14319B
- FACT utils-auto-prepended: PASS - empty-header kernel sees Limits<float>::max (=inf) from utils.h
- FACT absolute-include-resolves: PASS - header '#include "/Users/shivamgarg/dev/metal-autotune/spikes/out/spike08_tiny.h"' compiled and computed 3x+1 correctly
- FACT nested-relative-include-fails: PASS - compile failed as plan claims: 'mlx/backend/metal/kernels/steel/gemm/loader.h' file not found
- FACT flattened-utils-vs-prelude: INFO - re-inlining utils.h clashes with the auto-prepended copy (redefinition of 'bfloat16_to_uint16'); the flattener must skip utils.h and its transitive includes
- FACT header-trailing-newline: INFO - header ending in a line comment with no trailing newline swallows the generated kernel signature and fails to compile (mlx/backend/metal/kernels/utils.h:2415:23: error: 'buffer' attribute only applies to parameters, global constant variables, and non-static data members); the flattener must emit a trailing newline
- FACT include-flattener-feasible: PASS - flattened steel/gemm/gemm.h (9 files, 54592 chars, prelude skipped) compiles as header; sizeof(mlx::steel::GEMMParams)=72

**Deviations from the plan:** Three findings beyond the plan's text, none contradicting its core claims. (1) Size: plan 5.9 says "~4MB of MSL under .../kernels/"; measured 0.92MB under kernels/, with ~3.3MB for the whole include tree, so the 4MB figure describes the whole include dir, not the MSL subtree. (2) The auto-prepended utils.h prelude (plan 5.10, confirmed) means a flattener cannot naively inline utils.h or its five transitive includes (bf16.h, bf16_math.h, complex.h, defines.h, logging.h): doing so fails with redefinition errors, so stitch.py's flattener needs a skip set for the prelude. (3) A flattened header that ends in a line comment without a trailing newline comments out the generated [[kernel]] signature (MLX concatenates header and signature with no separator) and fails to compile with misleading errors attributed to utils.h; the flattener must terminate output with a newline. Nested quoted includes in the shipped tree are uniformly repo-relative ("mlx/backend/metal/kernels/..."), resolvable against site-packages/mlx/include/, and angle-bracket includes are all Metal system headers that need no handling.

## spike_09_compile: mx.compile caching and baseline dry run

- FACT mlx-version: INFO - mlx 0.32.2, device Device(gpu, 0)
- FACT compiled-callable-stale-after-swap: PASS - after child swap 2x->3x: eager step gives [3.0, 6.0, 9.0, 12.0], the already-compiled callable gives [2.0, 4.0, 6.0, 8.0] (pre-swap graph=True)
- FACT fresh-compile-same-callable-reflects-swap: FAIL - mx.compile(step) again (old compiled object still alive) gives [2.0, 4.0, 6.0, 8.0], want post-swap [3.0, 6.0, 9.0, 12.0]
- FACT fresh-compile-after-dropping-old: INFO - after deleting both old compiled objects, mx.compile(step) gives [3.0, 6.0, 9.0, 12.0] (sees the swap=True)
- FACT fresh-compile-new-closure: INFO - compiling a newly defined closure over the same model gives [3.0, 6.0, 9.0, 12.0] (sees the swap=True)
- FACT attr-counter-frozen: INFO - __call__ increments a python counter: compiled call1=[3.0, 5.0] call2=[3.0, 5.0] (identical=True), count attr=1 after 2 calls (body ran only at trace)
- FACT array-attr-reassign-frozen: INFO - gain reassigned 2.0->5.0 between calls: compiled returns [3.0, 5.0] (old array baked into the trace=True)
- FACT per-shape-retrace: PASS - 5 calls over shapes [(2,3),(2,3),(4,3),(4,3),(2,3)] ran the python body for [(2, 3), (4, 3)] (want once per distinct shape)
- FACT plain-vs-compiled-step: INFO - 4-layer MLP dim 512 batch 32: plain median 0.251 ms/step, compiled median 0.245 ms/step, median paired delta +2.4% (positive means compiled faster), warmed in 30/30 samples over 40 interleaved pairs
- FACT nested-compile: INFO - mx.compile(outer) calling a compiled inner: correct output=True, inner python body ran 1 more time(s) during the outer trace (shape already in the inner cache), outer body ran 1 time(s) over 2 calls

**Deviations from the plan:** 1) The plan's claim that a fresh mx.compile of the same callable reflects a module swap FAILED on mlx 0.32.2: the compile cache is keyed on the function object's identity and erased only when a compiled wrapper object is destroyed, so mx.compile(step) again while any old compiled object is alive returns the stale pre-swap graph. The remedy law 10 needs is stronger than 'call mx.compile again': either compile a newly defined function object (verified to see the swap), or drop every old compiled object first (verified: after del + gc, recompiling the same function retraces). 2) Nested compile is not opaque at trace time: the inner compiled fn's python body reruns while the outer fn is being traced (inlined into the outer trace), even for a shape already in the inner cache; this matters for any tracer logic assuming compiled sections never re-execute python. 3) Timing dry run: compiled was ~2% FASTER than plain on this net (the plan's motivating history had it ~4% slower once), and absolute step times swung ~2x between two runs while the paired median delta stayed ~+2%, confirming the plan's per-job empirical baseline choice and law 8. State freezing behaved as the plan assumes: python attribute writes happen only at trace time (counter stuck at 1), and both python scalars and reassigned mx.array attributes are baked into the trace as constants.

## spike_10_measurement: Measurement floors and peaks

- FACT timing-recipe: INFO - sync/perf_counter/eval/sync on a depth-32 1024x1024 fp32 matmul chain: median 24.75 ms per sample over 5 samples (MLX_MAX_OPS_PER_BUFFER=unset)
- FACT lazy-graph-build: PASS - graph construction without eval took 0.049 ms vs 24.75 ms evaluated; a timed loop must eval its outputs
- FACT pacing-clock-ramp: INFO - after a 3w pacing idle the first sample reads 1.59x steady state (positions >1% slow: [0]); paced chunks must warm 2 samples before timing, refines law 2
- FACT aa-null-cool: INFO - median paired delta -0.029% of median, IQR 0.393% (9 ABAB pairs in warmed chunks at 25.13 ms/sample); the IQR is the session floor
- FACT thermal-hot-latency: INFO - quick mode, load too short for real thermal signal; after 5s continuous work latency is +0.8% vs cool (25.34 ms vs 25.13 ms)
- FACT thermal-hot-floor: INFO - quick mode, load too short for real thermal signal; A/A floor hot/cool = 1.85x (IQR 0.729% vs 0.393%)
- FACT thermal-recovery: INFO - quick mode, load too short for real thermal signal; after 2s idle latency is +0.4% vs cool, floor 0.80x cool (IQR 0.314%)
- FACT duty-paced-drift: INFO - quick mode, blocks too short to accumulate heat; paced block (idle 3w after each chunk), 9 chunks over 9s wall: last-quarter/first-quarter chunk median = 1.0021 at 25.20 ms/sample
- FACT duty-unpaced-drift: INFO - quick mode, blocks too short to accumulate heat; unpaced back-to-back block, same work over 2s wall: last-quarter/first-quarter chunk median = 1.0120 at 25.42 ms/sample
- FACT warm-until-stable: INFO - first-ever kernel: cap 30 hit with no two consecutive timings within 1% (1% of 1.36 ms is under the dispatch jitter); first call 85.83 ms is 63.0x steady state 1.363 ms
- FACT mem-pressure-floor: INFO - with 6 GiB extra resident (6.1 GiB active) A/A floor = 4.74x baseline (IQR 0.898% vs 0.189%), latency -1.0%
- FACT peak-bandwidth: INFO - eager add on 256 MiB arrays (reads 2, writes 1): running max 95.5 GB/s over 6 reps
- FACT peak-flops-fp16: INFO - 3072x3072 matmul: running max 3272 GFLOP/s over 6 reps
- FACT peak-flops-bf16: INFO - 3072x3072 matmul: running max 3303 GFLOP/s over 6 reps
- FACT peak-flops-fp32: INFO - 3072x3072 matmul: running max 2833 GFLOP/s over 6 reps

**Deviations from the plan:** Four findings the plan does not state. (1) New platform fact refining law 2: after a 3w pacing idle the GPU runs at ramped-down clocks, so the first sample of a paced chunk reads 1.5-1.6x steady state (positions 1+ are clean). Pacing at single-sample granularity therefore biases every pair (measured pair deltas of +10 to +74%) and inflates the A/A floor to ~20% IQR; pacing must be chunk-granular with ~2 unmeasured warm samples per chunk, which brings the floor to 0.2-0.6% IQR. The spike bakes this chunk design into every timed path and reports it as FACT pacing-clock-ramp. Law 3's warm-until-stable is per-shape; this warmup is needed per chunk, every chunk. (2) Law 3's 1% consecutive-agreement rule does not converge on a ~1ms kernel: 1% is ~10us, under dispatch jitter, so the cap of 30 is hit. The rule needs an absolute epsilon floor for small kernels; the JIT story itself is confirmed (first call 63-95x steady state, dominated by ~90ms Metal compile). (3) Eager MLX does not fuse elementwise chains: 2*x+y runs as two kernels (5N traffic, reads 98 GB/s at 5N but would misreport as 59 GB/s if assumed 3N). The peaks microbench must count traffic per actual eager kernels; the spike uses a single add kernel (3N) measuring ~95 GB/s. (4) The quick-mode memory-pressure floor is unstable at 9 pairs (0.53x one run, 4.74x the next); law 9's claim needs --full's 30 pairs to grade. Thermal facts (laws 1-2, the 40-55% inflation claim) are deliberately weak in quick mode and wait for --full. Peak numbers for the roofline on this M4: ~95 GB/s bandwidth, 3.3 TFLOP/s fp16/bf16, 2.8 TFLOP/s fp32.


### spike_10 full mode (serial run, quiet machine)

- FACT timing-recipe: INFO - sync/perf_counter/eval/sync on a depth-32 1024x1024 fp32 matmul chain: median 24.75 ms per sample over 5 samples (MLX_MAX_OPS_PER_BUFFER=unset)
- FACT lazy-graph-build: PASS - graph construction without eval took 0.068 ms vs 24.75 ms evaluated; a timed loop must eval its outputs
- FACT pacing-clock-ramp: INFO - after a 3w pacing idle the first sample reads 1.43x steady state (positions >1% slow: [0]); paced chunks must warm 2 samples before timing, refines law 2
- FACT aa-null-cool: INFO - median paired delta -0.047% of median, IQR 0.400% (30 ABAB pairs in warmed chunks at 25.11 ms/sample); the IQR is the session floor
- FACT thermal-hot-latency: INFO - after 120s continuous work latency is +3.3% vs cool (25.93 ms vs 25.11 ms)
- FACT thermal-hot-floor: INFO - A/A floor hot/cool = 1.11x (IQR 0.445% vs 0.400%)
- FACT thermal-recovery: INFO - after 90s idle latency is +0.7% vs cool, floor 3.15x cool (IQR 1.259%)
- FACT duty-paced-drift: INFO - paced block (idle 3w after each chunk), 93 chunks over 100s wall: last-quarter/first-quarter chunk median = 0.9918 at 25.29 ms/sample
- FACT duty-unpaced-drift: INFO - unpaced back-to-back block, same work over 24s wall: last-quarter/first-quarter chunk median = 1.0317 at 25.65 ms/sample
- FACT warm-until-stable: INFO - first-ever kernel: 3 calls until two consecutive timings agree within 1%; first call 102.75 ms is 143.9x steady state 0.714 ms
- FACT mem-pressure-floor: INFO - with 6 GiB extra resident (6.1 GiB active) A/A floor = 0.70x baseline (IQR 0.326% vs 0.463%), latency +0.2%
- FACT peak-bandwidth: INFO - eager add on 256 MiB arrays (reads 2, writes 1): running max 92.8 GB/s over 18 reps
- FACT peak-flops-fp16: INFO - 3072x3072 matmul: running max 3305 GFLOP/s over 18 reps
- FACT peak-flops-bf16: INFO - 3072x3072 matmul: running max 3313 GFLOP/s over 18 reps
- FACT peak-flops-fp32: INFO - 3072x3072 matmul: running max 2850 GFLOP/s over 18 reps

**Full-mode reading:** this M4 is thermally mild: 120s of sustained load raised latency
only +3.3% and the A/A floor 1.11x (the plan's 40-55% motivating history came from other
machines; the pacing laws stay because they are protective and cheap). Cool A/A floor is
~0.4% IQR at 25ms/sample over 30 pairs. First-ever kernel compile is ~100ms (144x steady
state). The 6 GiB memory-pressure probe did not inflate the floor on this machine.
Roofline peaks for this chip: 92.8 GB/s bandwidth, 3305/3313/2850 GFLOP/s fp16/bf16/fp32.


## Post-spike findings from module builds

Verified by experiment while building M5-M8 (each has a test in the named file):

- An infinite-loop Metal kernel with volatile device accesses hangs mx.eval
  indefinitely with no OS-side error; a side-effect-free infinite loop is compiled
  away entirely. The parent wall timeout is therefore the only hang defense, and the
  GPU recovers cleanly after SIGKILL (tests/test_sandbox.py).
- Compile-error line offsets can be measured per kernel by prepending an
  `#error` probe line; the report lands at offset+1 without changing the generated
  signature (autotuner/sandbox/worker.py).
- Pool saturation is observable: a no-write kernel's recycled output buffer reads
  back 100% NaN after saturating with NaN-filled buffers (tests/test_sandbox.py).
- Shader-validation stderr reporting is asynchronous: scan child stderr after exit,
  not during the run (autotuner/sandbox/protocol.py).
- metal::precise:: tanh/cos/sqrt/rsqrt are bitwise-equal to the library kernels,
  extending spike_04's exp/sin fact; library sigmoid differs from the 1/(1+exp(-x))
  composition by 1 ulp, so sigmoid is tolerance-only (tests/test_scaffold.py).
- mlx Module.update_modules with strict=False silently no-ops on numeric dict keys;
  install code must never rely on non-strict mode (tests/test_platform_swap.py).

- mlx.nn ships EVERY activation pre-compiled at import time
  (@partial(mx.compile, shapeless=True) on silu, gelu, softmax, and 22 more).
  A pre-existing compiled object called while recording leaks its compile-trace
  placeholder arrays into the record unless recording is suppressed around the
  call (caught by the completeness check on the first real decoder fixture).
  The tracer wraps each one in a proxy that runs it with recording suppressed
  and records one call named by import path (compiled:mlx.nn.silu); the spec's
  rule holds, no region includes the activation, and a scope that calls it
  stays replayable because the wrapper calls the same path
  (tests/test_tracer.py, tests/test_bind.py, tests/test_llama_ish.py).
- Llama 3 8B (4-bit, mlx_lm) traces completely: 739 nodes, no completeness
  aborts, no mid-record evaluation. Copy grouping lands per the identity rule
  (per-layer stretches at copies=32, a 22-op attention+MLP stretch at
  copies=31, the quantized_matmul singleton at copies=225). mlx_lm's llama MLP
  routes silu(gate)*up through a module-level compiled helper created at
  mlx_lm import; it records as one call named compiled:mlx_lm.models.activations.swiglu
  per layer, so the fused kernel the model really runs is never priced as
  three separate ops, and the scopes around it stay replayable
  (spikes/llama8b_trace.py, before the import-path naming).

## Design adjustments recorded from these spikes

These changed IMPLEMENTATION_PLAN.md (each edit is tagged "M0 spike"):

1. Validate mode (plan section 8) must set `MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1`
   alongside `MTL_SHADER_VALIDATION=1` and parse child stderr for "Invalid device
   load"/"Invalid device store": validation alone zerofills silently and never faults.
2. Law 10's remedy is stronger than "call mx.compile again": the compile cache is keyed
   on function-object identity and survives while any old compiled object is alive, so
   after a swap the harness compiles a newly defined closure (or drops every old
   compiled object first).
3. The metal_kernel compile-error line offset is per kernel (grows with the generated
   signature), so the harness computes it per kernel rather than subtracting a constant.
4. Kernel names must be valid C identifiers; construction and call accept a bad name
   silently and the failure surfaces only at probe eval (static checks gate this).
5. The include flattener must skip the auto-prepended prelude (utils.h and its transitive
   includes) and terminate output with a newline.
6. Pacing (law 2) is chunk-granular: after each idle the GPU runs at ramped-down clocks
   and the first sample reads 1.5-1.6x slow, so every chunk takes ~2 unmeasured
   ramp-warm samples before timing. Warm-until-stable (law 3) needs an absolute epsilon
   floor; 1% of a ~1ms kernel is under dispatch jitter.
7. Generated bind wrappers must subclass nn.Module (a plain-object wrapper drops the
   subtree from parameters()/named_modules()), and a delegating __getattr__ must raise
   AttributeError before its wrapped attribute is set.
8. For transcendentals, library-bit fidelity comes from metal::precise:: namespacing,
   not math_mode; naive lowering uses precise:: where it must match library bits.
9. mlx nn.Module reserves `state` (property, no deleter): no fixture or model may
   assign self.state.
10. nn.Linear(bias=False) runs `x @ W.T` through array.__matmul__, never mx.matmul:
    module-level patching alone misses it; the dunder patches are load-bearing.

## Environment hazard observed live: silently degraded GPU (2026-08-31)

During the first real 8B job this M4 delivered ~1/10th of its own measured peaks
from the previous day, within the same boot session (uptime 57 days): 9.4 GB/s
and 0.30 TFLOP/s fp16 against the ~95 GB/s and 3.3 TFLOP/s above, with a fixed
~8 ms floor per eval round trip. Stable to 1%, so it is a state, not jitter.
Ruled out: harness leakage (a bare model in a fresh process was equally slow),
load, memory pressure and swap, Low Power Mode, thermals, power source, display
sleep. Cause unresolved at time of writing; reboot pending. Full findings and
reproduction scripts: work-2026-08-31-halted/.

What this changes: the pinned physics test (test_peaks_are_physically_sane)
failed loudly and correctly, and the harness now also checks its own job-start
peaks against the same plausibility floors (measure/peaks.py implausible()) and
logs an env_warning row plus a console warning, so a degraded machine is named
in-run instead of silently pricing every roofline against a sick GPU.

Addendum (2026-08-31, post-reboot): the cause is very likely NOT a stuck power
state. Ten minutes after the reboot, with the CPU idle, swap at zero, and no job
running, the GPU reads "Device Utilization % = 100" in IOAccelerator's
PerformanceStatistics: a background process is saturating the GPU. That explains
every prior observation, including the ones a power state couldn't: the stable
throughput division (fair timeslicing against a saturated queue), the fixed ~8 ms
eval round-trip floor (our command buffers queueing behind another process's
work), immunity to CPU load, memory, display, and power source, and the "recovery"
right after reboot that decayed within minutes (the daemon resumes its backlog).
Suspects present and GPU-invisible to CPU metrics: photoanalysisd, mediaanalysisd,
mlhostd, mds_stores. A reboot does not fix this machine; the GPU-idle check does:
probe utilization before any job, and treat healthy peak probes as the gate.
Check `ioreg -r -c IOAccelerator -d 4 | grep Utilization` reads ~0 at idle first.

## Scaffold coverage build (2026-08-31): quantized matmul, two ways

- Naive lowering handles mx.quantized_matmul (affine, transpose=True, bits 4/8,
  any supported group size): per-group factored fp32 accumulation, one packed
  uint32 load per 8 weights. On the real 8B decode GEMVs it sits inside the
  fp16 preserving tolerance (max abs diff 0.002 at K=14336) and inside the 10x
  watchdog (8.1-8.3x vs library). A region composing TWO quantized levels
  compounds reassociation past fp16 elementwise tolerance; such regions are
  assoc-changing territory (tests/test_scaffold.py pins the per-level bound).
- Impl-level MSL stitching works: mlx 0.32.2 ships its full Metal source tree
  in the wheel (include/mlx/backend/metal/kernels, 90 headers), and a
  metal_kernel body can call the wheel's qmv_fast_impl/qmv_impl directly after
  flattening quantized.h behind steel/gemm/gemm.h. Two enabling facts, both
  pinned in tests/test_stitch.py: the auto-injected inN_shape/inN_ndim buffers
  are constant address space and bind to const constant int& parameters, and
  simdgroup_index_in_threadgroup/thread_index_in_simdgroup are auto-injected.
  The library's qmv dispatch rule (fast when N % 8 == 0 and K % (2 * pack * 32)
  == 0) was mirrored and verified bitwise across an N/K sweep; stitched output
  is bitwise-identical to the library on every M=1 decode shape at 0.97-1.0x
  library speed. scaffold.build_scaffold prefers stitch for a lone
  quantized_matmul region and falls back to naive lowering.
- The fanless M4 thermally throttles ~4x after 10-20 minutes of sustained GPU
  work and recovers with ~20 minutes of idle cooling; reboots only ever helped
  by enforcing idle time. tests/conftest.py gates clock-sensitive tests on a
  quick bandwidth probe (floor 30 GB/s), the same floor as the harness's own
  in-run env_warning.

## spike_11_compile_patched: mx.compile over patched models and replays (2026-09-01)

- A compiled closure over a model holding a generated wrapper (with its shape
  guard) and a custom metal kernel compiles, and its outputs are bitwise equal
  to the plain call. Pinned: tests/test_platform_env_and_compile.py.
- A compiled closure built before a module swap keeps running the old graph
  after the swap; a fresh closure sees the new module. So every timed pass of
  the patched model comes from a closure built after the last swap
  (loop._step_fn).
- A compiled replay of a region's recorded ops equals the plain replay bitwise
  for an elementwise chain, in fp32 and fp16. Pinned in the same file.
- Compile halves the planted-win fixture's step (0.80 to 0.40 ms) and its
  8-op chain replay (0.81 to 0.40 ms): mx.compile already fuses an elementwise
  chain into one kernel, so under the compiled baseline that fusion is no win.
- Declared compile state (mx.compile inputs= and outputs=) is swapped inside
  the dict or list handed over, so a function must read the state through
  that container. State reached through a bare variable or an object
  attribute, which is where mlx_lm's KVCache keeps its arrays, raises
  "uncaptured inputs" when declared, and an undeclared compiled call leaves
  the array holding a tracer with no primitive, killing the model for every
  later call. Seen live on the 2026-09-01 23:48 Qwen decode run; pinned in
  tests/test_platform_env_and_compile.py. The harness therefore never
  compiles a step whose trace shows Python-retained arrays.

## Independent launches overlap (2026-09-02, work-2026-09-02-0028-est)

Metal runs independent kernels in one command buffer side by side. Forty
unchained launches of a one-threadgroup matvec read 0.031 ms per call hot and
0.083 cold; the same launches chained so each waits for the last read 0.329
and 0.375, and 0.36 is what the kernel cost per copy inside the model. MLX's
own matmul fills the GPU alone and barely moves (0.053 to 0.063 hot, 0.073 to
0.081 cold). Pinned in tests/test_measure.py (test_chained_launches_do_not_overlap).
Every timed loop in the harness now chains its passes and rotates a
cache-defeating working set (measure/clocks.py).

## spike_12_stream_probe: dependent kernels pay launch and stream in series (2026-09-02)

Measured on the M4 (10 GPU cores, 24 GB) with the clock laws (chained, cache-cold,
paired in one window), `spikes/spike_12_stream_probe.py`, and the open-time clocks
of work-2026-09-02-1054-est.

- **FACT launch-plus-stream.** MLX's bf16 matvec over a 2 MB matrix takes 29 to 32 us
  chained, against 22 us of stream at the measured 94.7 GB/s and a 7.1 us launch: the
  sum, not the larger of the two. 4 MB: 52 us measured, 44 + 7 arithmetic. 6 MB: 70
  measured, 66 + 7. The computed roofline `max(T_mem, T_compute, T_launch)` overstates
  headroom by a third at these sizes; the harness now measures the bytes-and-launch
  floor with a probe (`autotuner/measure/probe.py`) beside every region clock.
- **FACT matvec-at-the-wire.** Paired against the one-launch stream probe in one
  window, MLX's matvec reads 1.00 to 1.04x the probe at 2 MB, 1.01 to 1.04x at 4 MB,
  1.00 to 1.17x at 6 MB (five runs, stability 0.8 to 0.97). Separate measurements of
  the same two loops taken a minute apart read 0.6x to 1.5x on a GPU another process
  had at 89%, which is the case for the paired ratio. Pinned by
  `tests/test_measure.py::test_mlx_matvec_sits_near_the_stream_floor` (0.9x to 1.5x,
  health-gated).
- **FACT reductions-stream-slower.** `mx.sum(w, axis=1)` unchained reads 58 to 68 GB/s
  over the same matrices, slower than the matvec; MLX's own reduce is not a floor.
- **FACT constant-under-8.** `mx.fast.metal_kernel` binds an input with fewer than 8
  elements, whatever its dtype, in Metal's `constant` address space (8 or more:
  `device`); a `device` pointer cast to it fails to compile. The probe reads no input
  under 8 elements. Pinned by
  `tests/test_measure.py::test_stream_probe_runs_on_odd_shapes_and_small_inputs`,
  which reads an 8-element input through the cast.
- **FACT rms_norm-under-the-launch-figure.** MLX's rms_norm over 1024 elements runs
  in 4.7 to 5.9 us chained, under the 7.1 us the launch probe (a chain of scalar
  adds) reports per kernel, and 1.08x the stream probe over the same boundary. The
  computed roofline put its library under its own limit and reported 1.3 to 1.6x
  headroom on two such regions in the 1054 run.

