import { useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function App() {
  const [file, setFile] = useState(null);
  const [docInfo, setDocInfo] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_URL}/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload failed");
      setDocInfo(data);
    } catch (err) {
      alert(err.message);
    } finally {
      setUploading(false);
    }
  };

  const sendMessage = async (questionText) => {
    const q = questionText.trim();
    if (!q || streaming) return;

    const updated = [...messages, { role: "user", content: q }, { role: "assistant", content: "" }];
    setMessages(updated);
    setInput("");
    setStreaming(true);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });

        setMessages((prev) => {
          const copy = [...prev];
          copy[copy.length - 1] = {
            ...copy[copy.length - 1],
            content: copy[copy.length - 1].content + chunk,
          };
          return copy;
        });
      }
    } catch (err) {
      setMessages((prev) => {
        const copy = [...prev];
        copy[copy.length - 1].content = `Error: ${err.message}`;
        return copy;
      });
    } finally {
      setStreaming(false);
    }
  };

  return (
    <div style={{ display: "flex", height: "100vh", width: "100vw", fontFamily: "system-ui, sans-serif", background: "#0f172a", color: "#f8fafc" }}>
      {/* Sidebar */}
      <aside style={{ width: 300, padding: 24, background: "#1e293b", borderRight: "1px solid #334155", display: "flex", flexDirection: "column", gap: 16 }}>
        <h2 style={{ margin: 0, fontSize: 20 }}>📄 RAG Studio</h2>
        <p style={{ margin: 0, fontSize: 13, color: "#94a3b8" }}>FastAPI + Gemini 3.5 + ChromaDB</p>

        <input
          type="file"
          accept=".pdf"
          onChange={(e) => setFile(e.target.files[0])}
          style={{ fontSize: 13, color: "#cbd5e1" }}
        />

        <button
          onClick={handleUpload}
          disabled={!file || uploading}
          style={{ padding: "10px 14px", borderRadius: 8, border: "none", background: "#3b82f6", color: "white", fontWeight: 600, cursor: "pointer" }}
        >
          {uploading ? "Indexing PDF..." : "Process & Index PDF"}
        </button>

        {docInfo && (
          <div style={{ padding: 12, borderRadius: 8, background: "#0f172a", fontSize: 13, border: "1px solid #334155" }}>
            <div style={{ fontWeight: 600, color: "#38bdf8" }}>{docInfo.filename}</div>
            <div style={{ color: "#94a3b8", marginTop: 4 }}>
              {docInfo.pages} pages • {docInfo.chunks} chunks indexed
            </div>
            <button
              onClick={() => sendMessage("Summarize this PDF")}
              disabled={streaming}
              style={{ marginTop: 12, width: "100%", padding: "8px", borderRadius: 6, border: "1px solid #475569", background: "#1e293b", color: "#f8fafc", cursor: "pointer" }}
            >
              ✨ Summarize Full PDF
            </button>
          </div>
        )}
      </aside>

      {/* Main Chat Area */}
      <main style={{ flex: 1, display: "flex", flexDirection: "column" }}>
        <div style={{ flex: 1, padding: 24, overflowY: "auto", display: "flex", flexDirection: "column", gap: 14 }}>
          {messages.length === 0 && (
            <div style={{ margin: "auto", textAlign: "center", color: "#64748b" }}>
              <h3>Upload a PDF to start asking grounded questions</h3>
              <p>Supports page-cited RAG search and full-document summaries.</p>
            </div>
          )}
          {messages.map((m, i) => (
            <div
              key={i}
              style={{
                alignSelf: m.role === "user" ? "flex-end" : "flex-start",
                maxWidth: "75%",
                padding: "12px 16px",
                borderRadius: 12,
                background: m.role === "user" ? "#2563eb" : "#1e293b",
                whiteSpace: "pre-wrap",
                lineHeight: 1.5,
              }}
            >
              {m.content || "..."}
            </div>
          ))}
        </div>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            sendMessage(input);
          }}
          style={{ padding: 16, borderTop: "1px solid #334155", display: "flex", gap: 10 }}
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about your PDF..."
            style={{ flex: 1, padding: "12px 16px", borderRadius: 8, border: "1px solid #334155", background: "#1e293b", color: "white", outline: "none" }}
          />
          <button
            type="submit"
            disabled={streaming || !input.trim()}
            style={{ padding: "12px 20px", borderRadius: 8, border: "none", background: "#3b82f6", color: "white", fontWeight: 600, cursor: "pointer" }}
          >
            Send
          </button>
        </form>
      </main>
    </div>
  );
}