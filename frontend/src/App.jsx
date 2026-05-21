import { useState, useRef, useEffect } from "react";
import "./App.css";

const BASE_URL = "http://localhost:8000";

export default function App() {
  const [messages, setMessages]     = useState([]);
  const [input, setInput]           = useState("");
  const [loading, setLoading]       = useState(false);
  const [loadingMsg, setLoadingMsg] = useState("");
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loadingMsg]);

  async function ask() {
    if (!input.trim() || loading) return;

    const question = input.trim();
    setInput("");
    setMessages(p => [...p, { role: "user", text: question }]);
    setLoading(true);
    setLoadingMsg("⚙️ Analysing query intent...");   // Stage 1

    try {
      // ── Call 1: Get intent instantly ────────────────────────────────────────
      const intentRes  = await fetch(`${BASE_URL}/api/v1/intent`, {
        method : "POST",
        headers: { "Content-Type": "application/json" },
        body   : JSON.stringify({ question })
      });
      const intentData = await intentRes.json();
      const intent     = intentData.intent || "legal_query";
      const isSituation = intent === "situation_based";

      // Stage 2 — show detected message, stays visible while agent runs
      setLoadingMsg(
        isSituation
          ? "⚖️ Situation Based Query detected — fetching legal advice..."
          : "📋 Legal Query detected — searching legal database..."
      );

      // ── Call 2: Get answer (takes time, detected message stays) ─────────────
      const answerRes  = await fetch(`${BASE_URL}/api/v1/ask`, {
        method : "POST",
        headers: { "Content-Type": "application/json" },
        body   : JSON.stringify({ question })
      });
      const answerData = await answerRes.json();

      // Stage 3 — show answer with small badge
      setMessages(p => [...p, {
        role  : "bot",
        text  : answerData.answer,
        intent: answerData.intent || intent
      }]);

    } catch {
      setMessages(p => [...p, { role: "bot", text: "⚠️ Backend not reachable" }]);
    }

    setLoading(false);
    setLoadingMsg("");
  }

  function end() {
    setMessages([]);
    setInput("");
  }

  function intentLabel(intent) {
    if (!intent) return null;
    const isSituation = intent === "situation_based";
    return (
      <span style={{
        display      : "inline-block",
        marginBottom : "6px",
        padding      : "2px 10px",
        borderRadius : "12px",
        fontSize     : "0.75rem",
        fontWeight   : "600",
        background   : isSituation ? "#fff3cd" : "#d1ecf1",
        color        : isSituation ? "#856404" : "#0c5460",
        border       : `1px solid ${isSituation ? "#ffc107" : "#bee5eb"}`
      }}>
        {isSituation ? "⚖️ Situation Based Query" : "📋 Legal Query"}
      </span>
    );
  }

  return (
    <div className="container">
      <h1>🧑‍⚖️ Legal Advisor Chatbot</h1>

      <div className="card">
        <label>Ask your legal question:</label>
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === "Enter" && ask()}
          placeholder="e.g. What are duties of employer under POSH Act?"
        />
        <div className="buttons">
          <button onClick={ask} disabled={loading} className="primary">
            {loading ? "Getting Legal Advice..." : "Get Legal Advice"}
          </button>
          <button onClick={end} className="danger">
            End Conversation
          </button>
        </div>
      </div>

      {messages.length === 0 && !loading && (
        <div className="tips">
          <p><strong>💡 Tip:</strong> Try asking questions like:</p>
          <ul>
            <li>What are the conditions for a valid Hindu marriage?</li>
            <li>What are duties of an employer under POSH Act?</li>
            <li>What rights does a Hindu wife have?</li>
          </ul>
        </div>
      )}

      <div className="chat">
        {messages.map((m, i) => (
          <div key={i} className={m.role === "user" ? "user-msg" : "bot-msg"}>
            {m.role === "bot" && intentLabel(m.intent)}
            {m.role === "bot" && m.intent && <br />}
            {m.role === "bot" && m.text}
            {m.role === "user" && m.text}
          </div>
        ))}

        {/* Loading — changes from analysing → detected → disappears */}
        {loading && loadingMsg && (
          <div className="bot-msg" style={{ animation: "fadeIn 0.4s ease" }}>
            {loadingMsg}
          </div>
        )}

        <div ref={bottomRef} />
      </div>
    </div>
  );
}
