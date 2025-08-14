import React, { useState, useEffect } from "react";
import { useNodesState, useEdgesState, NodeChange } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import yaml from "yaml";
import { Sparkles, Database, Settings } from "lucide-react";

import { FileUpload } from "@/components/FileUpload";
import { TableModal } from "@/components/TableModal";
import { PipelineFlow } from "@/components/PipelineFlow";
import { YamlEditor } from "@/components/YamlEditor";
import { FunctionSearch } from "@/components/FunctionSearch";
import { ParameterSelector } from "@/components/ParameterSelector";

const API_BASE = (import.meta.env.VITE_API_BASE as string) || "/api";

// short name for node label
function shortName(full: string) {
  const parts = (full || "").split(".");
  return parts[parts.length - 1] || full;
}

export default function DataSciencePipelinePage() {
  const [nodes, setNodes, onNodesChangeBase] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [yamlText, setYamlText] = useState("nodes: {}");
  const [file, setFile] = useState<File | null>(null);
  const [preloaded, setPreloaded] = useState<any[]>([]);
  const [tableModal, setTableModal] = useState<{ open: boolean; table: any | null }>({
    open: false,
    table: null,
  });
  const [paramModal, setParamModal] = useState<{
    open: boolean;
    nodeId: string | null;
    funcDef: any | null;
    params: Record<string, any>;
    dependencies: string[];
  }>({
    open: false,
    nodeId: null,
    funcDef: null,
    params: {},
    dependencies: [],
  });

  // preload some pandas funcs (so we can auto-add read_csv)
  useEffect(() => {
    fetch(`${API_BASE}/pandas/search?query=read_`)
      .then((r) => r.json())
      .then((d) => Array.isArray(d?.functions) && setPreloaded(d.functions))
      .catch(() => {});
  }, []);

  // Auto-add read_csv when a file is uploaded and there are no nodes yet
  useEffect(() => {
    if (file && nodes.length === 0) {
      const readCsv = preloaded.find((f: any) => shortName(f.name) === "read_csv");
      if (readCsv) {
        createNodeWithParams(readCsv, { filepath_or_buffer: file.name });
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file, preloaded]);

  function nodeStyle() {
    return {
      background: "hsl(var(--card))",
      border: "2px solid hsl(var(--primary))",
      borderRadius: "12px",
      minWidth: "180px",
      minHeight: "60px",
      padding: "8px",
      boxShadow: "0 4px 20px hsl(var(--primary) / 0.3)",
    } as React.CSSProperties;
  }

  function addFunctionNode(func: any) {
    setParamModal({
      open: true,
      nodeId: null,
      funcDef: func,
      params: {},
      dependencies: [],
    });
  }

  function createNodeWithParams(func: any, selectedParams: Record<string, any>) {
    const id = `${shortName(func.name)}_${nodes.length}`;
    let dependencies: string[] = [];

    // if there is a previous node and this function takes df/self/left, wire it
    if (nodes.length > 0) {
      const prev = nodes[nodes.length - 1].id;
      const dfParam = func.params?.find((k: any) => ["df", "self", "left"].includes(k.name));
      if (dfParam && !selectedParams[dfParam.name]) {
        selectedParams[dfParam.name] = prev;
        dependencies = [prev];
      }
    }

    // auto-fill filename for read_* if available
    if (file && func.name.startsWith("read_")) {
      for (const k of ["filepath_or_buffer", "path_or_buf", "io", "file_path", "filepath"]) {
        if (func.params?.some((p: any) => p.name === k) && !selectedParams[k]) {
          selectedParams[k] = file.name;
          break;
        }
      }
    }

    // visual node
    setNodes((n) => [
      ...n,
      {
        id,
        data: {
          label: (
            <div className="flex items-center gap-2 px-3 py-2">
              <div className="p-1 rounded bg-primary/20">
                <Database className="w-3 h-3 text-primary" />
              </div>
              <span className="flex-1 truncate">{shortName(func.name)}</span>
              <button
                className="p-1 rounded bg-secondary/20 hover:bg-secondary/30"
                onClick={(e) => {
                  e.stopPropagation();
                  setParamModal({
                    open: true,
                    nodeId: id,
                    funcDef: func,
                    params: selectedParams,
                    dependencies,
                  });
                }}
                title="Edit Parameters"
              >
                <Settings className="w-3 h-3 text-secondary" />
              </button>
            </div>
          ),
        },
        position: { x: 220 + Math.random() * 100, y: 90 + 90 * nodes.length },
        style: nodeStyle(),
      },
    ]);

    // YAML update
    let parsed: any;
    try {
      parsed = yaml.parse(yamlText) || {};
    } catch {
      parsed = {};
    }
    if (!parsed.nodes) parsed.nodes = {};
    parsed.nodes[id] = { function: func.name, params: selectedParams, dependencies };
    setYamlText(yaml.stringify(parsed));
  }

  // keep edges in sync with YAML dependencies
  useEffect(() => {
    let parsed: any;
    try {
      parsed = yaml.parse(yamlText) || {};
    } catch {
      parsed = {};
    }
    const newEdges: any[] = [];
    if (parsed.nodes) {
      Object.entries(parsed.nodes).forEach(([id, value]: [string, any]) => {
        (value.dependencies || []).forEach((dep: string) => {
          newEdges.push({
            id: `${dep}->${id}`,
            source: dep,
            target: id,
            animated: true,
            style: { stroke: "hsl(var(--primary))", strokeWidth: 3 },
            type: "smoothstep",
          });
        });
      });
    }
    setEdges(newEdges);
  }, [yamlText, nodes.length, setEdges]);

  // 🔧 keep YAML in sync when nodes are removed from the canvas
  function onNodesChange(changes: NodeChange[]) {
    // first apply visual changes
    onNodesChangeBase(changes);

    // then delete from YAML if any node is removed
    const removed = changes.filter((c) => c.type === "remove").map((c: any) => c.id);
    if (removed.length === 0) return;

    let parsed: any;
    try {
      parsed = yaml.parse(yamlText) || {};
    } catch {
      parsed = {};
    }
    if (!parsed.nodes) return;

    removed.forEach((id) => {
      delete parsed.nodes[id];
      // also remove this id from any dependency lists
      Object.values(parsed.nodes).forEach((n: any) => {
        if (Array.isArray(n.dependencies)) {
          n.dependencies = n.dependencies.filter((d: string) => d !== id);
        }
      });
    });

    setYamlText(yaml.stringify(parsed));
  }

  // 👉 double-click: ask backend to execute YAML up to this node and return preview
// put this in DataSciencePipelinePage (replace the whole function)
async function handleNodeDoubleClick(e: any, node: any) {
  // don’t crash if no event
  e?.stopPropagation?.();

  const fd = new FormData();
  fd.append("yaml", yamlText);
  fd.append("preview_node", node.id);
  if (file) fd.append("file", file, file.name);

  try {
    const res = await fetch(`${API_BASE}/pipeline/run`, { method: "POST", body: fd });

    // ---- better error surfacing ----
    if (!res.ok) {
      // try JSON first (FastAPI often returns {"detail": "..."} or custom fields)
      let bodyText = "";
      try {
        const ct = res.headers.get("content-type") || "";
        if (ct.includes("application/json")) {
          const j = await res.json();
          const detail =
            j?.detail ??
            j?.error ??
            j?.message ??
            (Array.isArray(j) && j[0]?.msg) ??
            "";
          const tb = j?.traceback ?? j?.stack ?? "";
          bodyText = [detail, tb].filter(Boolean).join("\n\n");
        } else {
          bodyText = await res.text();
        }
      } catch {
        // fallback if parsing threw
        try { bodyText = await res.text(); } catch { bodyText = ""; }
      }
      const msg = `HTTP ${res.status} while previewing "${node?.id}":\n` +
                  (bodyText ? bodyText.slice(0, 2000) : "(no response body)");
      throw new Error(msg);
    }

    // OK → parse & show
    const data = await res.json(); // { columns: [...], rows: [...] }
    setTableModal({
      open: true,
      table: {
        columns: Array.isArray(data?.columns) ? data.columns : [],
        data: Array.isArray(data?.rows) ? data.rows.slice(0, 200) : [],
      },
    });
  } catch (err: any) {
    console.error("Preview error:", err);
    // show the *useful* part to the user
    const msg = String(err?.message || err || "Unknown error");
    alert(msg);
  }
}


  // used by YamlEditor "Run" button (demo: run the last node preview)
  async function runPipeline() {
    if (nodes.length === 0) {
      alert("Add at least one node first.");
      return;
    }
    await handleNodeDoubleClick({ stopPropagation() {} } as any, nodes[nodes.length - 1]);
  }

  function handleApplyParams(newParams: Record<string, any>) {
    const { nodeId, funcDef, dependencies } = paramModal;

    if (!nodeId) {
      // creating a new node
      createNodeWithParams(funcDef, newParams);
    } else {
      // updating existing YAML
      let parsed: any;
      try {
        parsed = yaml.parse(yamlText) || {};
      } catch {
        parsed = {};
      }
      if (!parsed.nodes) parsed.nodes = {};
      parsed.nodes[nodeId] = {
        function: funcDef.name,
        params: newParams,
        dependencies: dependencies || [],
      };
      setYamlText(yaml.stringify(parsed));
    }

    setParamModal({
      open: false,
      nodeId: null,
      funcDef: null,
      params: {},
      dependencies: [],
    });
  }

  return (
    <div className="pipeline-container">
      <div className="container mx-auto p-6">
        <header className="mb-8 text-center">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="p-3 rounded-xl bg-gradient-primary">
              <Sparkles className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-4xl font-bold glow-text">Tharavu Dappa</h1>
          </div>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Build data pipelines visually. Double-click any block to preview its real output.
          </p>
        </header>

        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 h-[80vh]">
          <YamlEditor yamlText={yamlText} onYamlChange={setYamlText} onRunPipeline={runPipeline} />

          <div className="pipeline-panel flex flex-col">
            <div className="p-6 border-b border-border/50">
              <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center">
                <FileUpload onFileSelected={setFile} selectedFile={file} />
                <FunctionSearch onSelectFunction={addFunctionNode} />
              </div>
            </div>
            <div className="flex-1 p-6">
              <PipelineFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onNodeDoubleClick={handleNodeDoubleClick}
              />
            </div>
          </div>
        </div>
      </div>

      <ParameterSelector
        open={paramModal.open}
        onClose={() => setParamModal({ ...paramModal, open: false })}
        funcDef={paramModal.funcDef}
        initialParams={paramModal.params}
        onApply={handleApplyParams}
      />

      <TableModal
        open={tableModal.open}
        onClose={() => setTableModal({ open: false, table: null })}
        table={tableModal.table}
      />
    </div>
  );
}
