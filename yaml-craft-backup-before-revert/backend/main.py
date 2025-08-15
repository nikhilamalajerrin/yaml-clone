"""
Tharavu Dappa Backend — Light index + Robust pipeline executor
- Keeps /pandas/search endpoints for FunctionSearch
- Adds /pipeline/run with param coercion & reference resolution
"""

import inspect
import importlib
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import yaml
from io import BytesIO
import ast  # <-- added for indexer parsing & literal_eval
import traceback

app = FastAPI(title="Tharavu Dappa Backend", version="3.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- utils --------------------

def _callable(x) -> bool:
    try:
        return callable(x) and not inspect.isclass(x)
    except Exception:
        return False

def _safe_sig(obj):
    try:
        return inspect.signature(obj)
    except Exception:
        return None

def get_function_signature(func: Any) -> Dict[str, Any]:
    sig = _safe_sig(func)
    params = []
    if sig:
        for name, p in sig.parameters.items():
            params.append({
                "name": name,
                "kind": str(p.kind),
                "required": p.default == inspect.Parameter.empty,
                "default": None if p.default == inspect.Parameter.empty else repr(p.default),
                "annotation": None if p.annotation == inspect.Parameter.empty else str(p.annotation),
            })
    return {
        "name": getattr(func, "__name__", "unknown"),
        "doc": inspect.getdoc(func) or "No documentation available",
        "params": params,
        "module": getattr(func, "__module__", "unknown"),
    }

def _add(functions: List[Dict[str, Any]], names: Set[str],
         obj: Any, suggestion: str, library: str, category: str, canonical: str):
    info = get_function_signature(obj)
    info["library"] = library
    info["category"] = category
    info["name"] = canonical
    functions.append(info)
    names.add(suggestion)

def _indexer_stubs_for_class(cls, cls_name: str) -> List[Dict[str, Any]]:
    """
    Auto-discover indexers (iloc/loc/at/iat) by inspecting a tiny dummy instance.
    We don't change your behavior; just make them searchable with synthetic params.
    """
    stubs = []
    dummy = None
    try:
        if cls_name == "DataFrame":
            dummy = pd.DataFrame({"_": [0]})
        elif cls_name == "Series":
            dummy = pd.Series([0])
        elif cls_name == "Index":
            dummy = pd.Index([0])
    except Exception:
        dummy = None

    if dummy is None:
        return stubs

    for attr in dir(dummy):
        if attr.startswith("_"):
            continue
        try:
            val = getattr(dummy, attr)
        except Exception:
            continue
        t = type(val)
        mod = getattr(t, "__module__", "")
        tname = getattr(t, "__name__", "")
        if "pandas.core.indexing" in mod and "Indexer" in tname:
            # synthetic signature for your Param UI
            params = [{"name": "self", "kind": "POSITIONAL_OR_KEYWORD", "required": True, "default": None, "annotation": None}]
            if "AtIndexer" in tname or attr in ("at", "iat"):
                params += [
                    {"name": "row", "kind": "POSITIONAL_OR_KEYWORD", "required": True, "default": None, "annotation": None},
                    {"name": "col", "kind": "POSITIONAL_OR_KEYWORD", "required": True, "default": None, "annotation": None},
                    {"name": "value", "kind": "POSITIONAL_OR_KEYWORD", "required": False, "default": None, "annotation": None},
                ]
            else:
                params += [
                    {"name": "rows", "kind": "POSITIONAL_OR_KEYWORD", "required": False, "default": ":", "annotation": None},
                    {"name": "cols", "kind": "POSITIONAL_OR_KEYWORD", "required": False, "default": ":", "annotation": None},
                ]
            stubs.append({
                "name": f"{cls_name}.{attr}",
                "doc": getattr(pd.DataFrame, attr, None).__doc__ or f"{cls_name}.{attr} indexer",
                "params": params,
                "module": "pandas.core.indexing",
                "library": "pandas",
                "category": cls_name,
            })
    return stubs

def _collect_light() -> Tuple[List[Dict[str, Any]], List[str]]:
    functions: List[Dict[str, Any]] = []
    suggestions: Set[str] = set()

    # pandas top-level
    for name in dir(pd):
        if name.startswith("_"): continue
        try:
            obj = getattr(pd, name)
        except Exception:
            continue
        if _callable(obj):
            _add(functions, suggestions, obj, name, "pandas", "pandas", name)

    # key pandas classes/methods
    for cls in filter(None, [getattr(pd,"DataFrame",None),
                             getattr(pd,"Series",None),
                             getattr(pd,"Index",None),
                             getattr(pd,"Categorical",None)]):
        cls_name = getattr(cls, "__name__", "PandasClass")
        for m in dir(cls):
            if m.startswith("_"): continue
            try:
                meth = getattr(cls, m)
            except Exception:
                continue
            if _callable(meth):
                _add(functions, suggestions, meth, m, "pandas", cls_name, f"{cls_name}.{m}")
                suggestions.add(f"{cls_name}.{m}")
        # auto-inject indexers for search (iloc/loc/at/iat)
        for stub in _indexer_stubs_for_class(cls, cls_name):
            functions.append(stub)
            suggestions.add(stub["name"])

    # pandas submodules (light)
    for sub in ("io", "plotting"):
        try:
            submod = getattr(pd, sub)
            for a in dir(submod):
                if a.startswith("_"): continue
                try:
                    obj = getattr(submod, a)
                except Exception:
                    continue
                if _callable(obj):
                    _add(functions, suggestions, obj, a, "pandas", f"pandas.{sub}", f"{sub}.{a}")
                    suggestions.add(f"{sub}.{a}")
        except Exception:
            pass

    # numpy top-level
    for a in dir(np):
        if a.startswith("_"): continue
        try:
            obj = getattr(np, a)
        except Exception:
            continue
        if _callable(obj):
            _add(functions, suggestions, obj, a, "numpy", "NumPy", a)

    # numpy submodules (light)
    for sub in ("linalg", "random", "fft"):
        try:
            submod = getattr(np, sub)
            for a in dir(submod):
                if a.startswith("_"): continue
                try:
                    obj = getattr(submod, a)
                except Exception:
                    continue
                if _callable(obj):
                    _add(functions, suggestions, obj, f"{sub}.{a}", "numpy", f"numpy.{sub}", f"{sub}.{a}")
                    suggestions.add(a)
        except Exception:
            pass

    seen = set()
    out = []
    for f in functions:
        key = (f.get("library"), f.get("name"))
        if key in seen: continue
        seen.add(key); out.append(f)
    return out, sorted(suggestions)

@lru_cache(maxsize=1)
def get_index() -> Tuple[List[Dict[str, Any]], List[str]]:
    return _collect_light()

# -------------------- function resolution --------------------

def get_callable_from_name(func_name: str):
    """Resolve a pandas/numpy function or pandas method by canonical/name."""
    # indexers are not callables: handled in pipeline_run (we keep your resolver unchanged)
    # module path (pandas.x.y or numpy.x.y)
    if func_name.startswith("pandas.") or func_name.startswith("numpy."):
        parts = func_name.split(".")
        for cut in range(len(parts), 0, -1):
            mod_name = ".".join(parts[:cut])
            try:
                mod = importlib.import_module(mod_name)
                obj = mod
                ok = True
                for p in parts[cut:]:
                    if not hasattr(obj, p):
                        ok = False; break
                    obj = getattr(obj, p)
                if ok and _callable(obj):
                    return obj
            except Exception:
                continue

    # short submodule form like "linalg.norm" or "plotting.scatter_matrix"
    if "." in func_name:
        head, tail = func_name.split(".", 1)
        if hasattr(np, head):
            sub = getattr(np, head)
            if hasattr(sub, tail):
                cand = getattr(sub, tail)
                if _callable(cand): return cand
        if hasattr(pd, head):
            sub = getattr(pd, head)
            if hasattr(sub, tail):
                cand = getattr(sub, tail)
                if _callable(cand): return cand

    # pandas top-level
    if hasattr(pd, func_name):
        cand = getattr(pd, func_name)
        if _callable(cand): return cand

    # pandas methods like "DataFrame.rename"
    pandas_classes = {
        "DataFrame": getattr(pd, "DataFrame", None),
        "Series": getattr(pd, "Series", None),
        "Index": getattr(pd, "Index", None),
        "Categorical": getattr(pd, "Categorical", None),
    }
    if "." in func_name:
        cls_name, meth = func_name.split(".", 1)
        cls = pandas_classes.get(cls_name)
        if cls and hasattr(cls, meth):
            cand = getattr(cls, meth)
            if _callable(cand): return cand

    # numpy top-level
    if hasattr(np, func_name):
        cand = getattr(np, func_name)
        if _callable(cand): return cand

    raise ValueError(f"Function '{func_name}' not found")

# -------------------- param coercion & reference resolution --------------------

def _looks_like_mapping_str(s: str) -> bool:
    # e.g. "A:B" or "A: B" but not JSON-like braces
    return (":" in s) and not (s.strip().startswith("{") and s.strip().endswith("}"))

def _looks_like_list_str(s: str) -> bool:
    # "a,b,c" or "[a,b]"
    st = s.strip()
    return "," in st or (st.startswith("[") and st.endswith("]"))

def _try_yaml_or_json_scalar(s: str) -> Any:
    # Try YAML parse on single-line scalars/lists/dicts
    try:
        v = yaml.safe_load(s)
        return v
    except Exception:
        return s

def _coerce_value(v: Any) -> Any:
    if isinstance(v, str):
        sv = v.strip()

        # Single-pair shorthand "OLD:NEW" → {"OLD":"NEW"}
        if _looks_like_mapping_str(sv) and ("\n" not in sv):
            left, _, right = sv.partition(":")
            if left and right:
                return {left.strip(): right.strip()}

        # List-ish "a,b" or "[a,b]" → ["a", "b"]
        if _looks_like_list_str(sv):
            if sv.startswith("[") and sv.endswith("]"):
                parsed = _try_yaml_or_json_scalar(sv)
                return parsed
            else:
                return [s.strip() for s in sv.split(",") if s.strip()]

        # Try YAML for booleans/numbers/null/inline dict/list
        parsed = _try_yaml_or_json_scalar(sv)
        return parsed
    return v

def coerce_params(params: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _coerce_value(v) for k, v in params.items()}

def extract_param_node_refs(params: Dict[str, Any]) -> Set[str]:
    """Find string values that look like node ids (to ensure they are executed)."""
    refs = set()
    for v in params.values():
        if isinstance(v, str):
            refs.add(v)
        elif isinstance(v, (list, tuple)):
            for x in v:
                if isinstance(x, str):
                    refs.add(x)
        elif isinstance(v, dict):
            for x in v.values():
                if isinstance(x, str):
                    refs.add(x)
    return refs

def resolve_param_references(params: Dict[str, Any], executed: Dict[str, Any]) -> Dict[str, Any]:
    """Replace any param values equal to node ids with the executed object."""
    def resolve(v):
        if isinstance(v, str) and v in executed:
            return executed[v]
        if isinstance(v, list):
            return [resolve(x) for x in v]
        if isinstance(v, tuple):
            return tuple(resolve(x) for x in v)
        if isinstance(v, dict):
            return {k: resolve(x) for k, x in v.items()}
        return v
    return {k: resolve(v) for k, v in params.items()}

def _normalize_axis(v: Any) -> Any:
    # VERY light normalization; doesn't change other behavior
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "columns", "column", "cols"): return 1
        if s in ("0", "index", "row", "rows"): return 0
    return v

# Indexer parsing (strings -> slice/list/int/label)
def _parse_indexer(v, numeric: bool):
    """
    Accepts:
      ":" -> slice(None)
      "a:b" / "a:b:c" -> slice (ints if numeric=True else labels)
      "0" -> int if numeric else label "0"
      "[0,1]" / "['A','B']" -> list via literal_eval
      plain strings -> label (or int if numeric=True)
    """
    if v is None:
        return slice(None)
    if isinstance(v, slice):
        return v
    if isinstance(v, (list, tuple, int)):
        return v

    if isinstance(v, str):
        s = v.strip()
        if s == "" or s == ":":
            return slice(None)
        # slice notations
        if ":" in s and not (s.startswith("{") or s.startswith("[")):
            parts = s.split(":")
            def _to_int(x):
                x = x.strip()
                return None if x == "" else (int(x) if numeric else x)
            if len(parts) == 2:
                a, b = parts
                return slice(_to_int(a), _to_int(b))
            if len(parts) == 3:
                a, b, c = parts
                return slice(_to_int(a), _to_int(b), _to_int(c))
        try:
            val = ast.literal_eval(s)
            return val
        except Exception:
            if numeric:
                try:
                    return int(s)
                except Exception:
                    pass
            return s
    return v

# -------------------- pipeline executor --------------------

@app.post("/pipeline/run")
async def pipeline_run(
    yaml: str = Form(...),
    preview_node: Optional[str] = Form(None),
    file: Optional[UploadFile] = None,
):
    try:
        spec = yaml and __import__("yaml").safe_load(yaml) or {}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")

    if not isinstance(spec, dict) or "nodes" not in spec or not isinstance(spec["nodes"], dict):
        raise HTTPException(status_code=400, detail="YAML must contain 'nodes' dict")

    nodes: Dict[str, Any] = spec["nodes"]
    executed: Dict[str, Any] = {}
    remaining = set(nodes.keys())

    # file bytes (for read_* convenience)
    uploaded_bytes = await file.read() if file else None

    # Execute until all done or we can't progress
    while remaining:
        made_progress = False

        for node_id in list(remaining):
            node_def = nodes[node_id]
            func_name = node_def.get("function")
            raw_params = dict(node_def.get("params", {}))
            deps = list(node_def.get("dependencies", []))

            # Coerce param values (strings → dict/list/bool/number when appropriate)
            params = coerce_params(raw_params)

            # Ensure implicit deps: any param referencing node ids must be executed first
            implicit_refs = extract_param_node_refs(params)
            all_deps = set(deps) | (implicit_refs & set(nodes.keys()))
            if any(d not in executed for d in all_deps):
                continue

            # Resolve references (turn "read_csv_0" into actual DataFrame)
            params = resolve_param_references(params, executed)

            # Light axis normalization only (safe)
            if "axis" in params:
                params["axis"] = _normalize_axis(params["axis"])

            # Receiver for methods (self/df/left)
            recv = None
            if "self" in raw_params:
                k = raw_params["self"]
                recv = executed.get(k) if isinstance(k, str) else k
                params.pop("self", None)
            elif "df" in raw_params:
                k = raw_params["df"]
                recv = executed.get(k) if isinstance(k, str) else k
                params.pop("df", None)
            elif "left" in raw_params and func_name.endswith(".merge"):
                k = raw_params["left"]
                recv = executed.get(k) if isinstance(k, str) else k
                params.pop("left", None)

            # read_* auto: feed uploaded file
            if uploaded_bytes is not None and isinstance(func_name, str) and func_name.startswith("read_"):
                for k in ["filepath_or_buffer", "path_or_buf", "io", "file_path", "filepath"]:
                    if k in params:
                        params[k] = BytesIO(uploaded_bytes)

            # ----- SPECIAL: DataFrame indexers (iloc/loc/at/iat) -----
            if isinstance(func_name, str) and func_name.startswith("DataFrame."):
                attr = func_name.split(".", 1)[1]
                try:
                    dummy = pd.DataFrame({"_":[0]})
                    if hasattr(dummy, attr):
                        # verify it's really an indexer
                        idxr = getattr(dummy, attr)
                        t = type(idxr)
                        if "pandas.core.indexing" in getattr(t, "__module__", "") and "Indexer" in getattr(t, "__name__", ""):
                            # need a receiver
                            if recv is None:
                                # also support df param if user used that name
                                df_recv = params.pop("df", None)
                                if isinstance(df_recv, str):
                                    df_recv = executed.get(df_recv)
                                recv = recv or df_recv
                            if recv is None:
                                raise HTTPException(status_code=400, detail=f"{func_name} requires a receiver (self/df)")

                            if attr in ("loc", "iloc"):
                                rows = _parse_indexer(params.pop("rows", ":"), numeric=(attr == "iloc"))
                                cols = _parse_indexer(params.pop("cols", ":"), numeric=(attr == "iloc"))
                                result = getattr(recv, attr)[rows, cols]
                            else:  # at / iat
                                row = _parse_indexer(params.pop("row", None), numeric=(attr == "iat"))
                                col = _parse_indexer(params.pop("col", None), numeric=(attr == "iat"))
                                if "value" in params:
                                    getattr(recv, attr)[row, col] = params["value"]
                                    result = recv
                                else:
                                    result = getattr(recv, attr)[row, col]

                            executed[node_id] = result
                            remaining.remove(node_id)
                            made_progress = True
                            if preview_node and node_id == preview_node:
                                return serialize_result(result)
                            continue
                except HTTPException:
                    raise
                except Exception as e:
                    tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                    raise HTTPException(status_code=500, detail=f"Error executing node '{node_id}' ({func_name}): {e}\n{tb}")

            # Resolve function/method/np callables (unchanged)
            try:
                func = get_callable_from_name(func_name)
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

            # Special-case top-level pd.merge(left, right, **kwargs)
            if func is pd.merge:
                left_obj = params.pop("left", None)
                right_obj = params.pop("right", None)
                if isinstance(left_obj, str):
                    left_obj = executed.get(left_obj)
                if isinstance(right_obj, str):
                    right_obj = executed.get(right_obj)
                if left_obj is None or right_obj is None:
                    raise HTTPException(status_code=400, detail=f"Node '{node_id}': pd.merge requires left and right")
                try:
                    result = func(left_obj, right_obj, **params)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error executing node '{node_id}' ({func_name}): {e}")
            else:
                try:
                    if recv is not None:
                        result = func(recv, **params)
                    else:
                        result = func(**params)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error executing node '{node_id}' ({func_name}): {e}")

            executed[node_id] = result
            remaining.remove(node_id)
            made_progress = True

            if preview_node and node_id == preview_node:
                return serialize_result(result)

        if not made_progress:
            raise HTTPException(status_code=400, detail="Pipeline has cyclic or unsatisfied dependencies.")

    # No preview requested → return last
    last_key = list(executed.keys())[-1]
    return serialize_result(executed[last_key])

# -------------------- result serialization --------------------

def serialize_result(result: Any):
    # EXACTLY your behavior: stringify to avoid nulls
    if isinstance(result, pd.DataFrame):
        return {"columns": list(result.columns), "rows": result.astype(str).values.tolist()}
    if isinstance(result, pd.Series):
        return {"columns": [result.name or "value"], "rows": [[str(v)] for v in result.values]}
    if isinstance(result, np.ndarray):
        return {"columns": ["value"], "rows": [[str(v)] for v in result.flatten()]}
    return {"columns": ["value"], "rows": [[str(result)]]}

# -------------------- search endpoints (unchanged) --------------------

@app.get("/")
async def root():
    return {"ok": True}

@app.get("/healthz")
async def health():
    return {"status": "ready"}

@app.get("/pandas/functions")
async def functions_all():
    funcs, _ = get_index()
    return {"functions": funcs, "total_count": len(funcs)}

@app.get("/pandas/suggest")
async def suggest(q: Optional[str] = ""):
    _, names = get_index()
    if not q:
        return {"suggestions": names[:50]}
    q = q.lower()
    starts = [n for n in names if n.lower().startswith(q)]
    contains = [n for n in names if q in n.lower() and n not in starts]
    starts.sort(key=lambda n: (len(n), n.lower()))
    contains.sort(key=lambda n: (len(n), n.lower()))
    return {"suggestions": (starts + contains)[:100]}

@app.get("/pandas/search")
async def search(query: str):
    funcs, _ = get_index()
    q = (query or "").strip().lower()
    if not q:
        return {"functions": funcs[:50], "total_count": len(funcs)}
    results = []
    for f in funcs:
        name = f["name"]
        plain = name.split(".")[-1].lower()
        doc = (f.get("doc") or "").lower()
        cat = (f.get("category") or "").lower()
        score = 0
        if name.lower() == q or plain == q: score += 120
        elif name.lower().startswith(q) or plain.startswith(q): score += 90
        elif q in name.lower() or q in plain: score += 70
        elif q in doc: score += 25
        elif q in cat: score += 15
        if score:
            g = dict(f); g["relevance_score"] = score; results.append(g)
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return {"functions": results[:50], "total_count": len(results)}

# -------------------- main --------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
