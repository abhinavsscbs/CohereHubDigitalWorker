import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Send,
  MessageSquare,
  Languages,
  Trash2,
  Download,
  FileText,
  ChevronDown,
  ChevronUp,
  Clock,
  Book,
  Sparkles,
  Loader2,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './App.css';

const API_BASE_URL = 'http://127.0.0.1:3000/api';
const SESSION_ID = 'user_session_' + Math.random().toString(36).substr(2, 9);

function App() {
  const [question, setQuestion] = useState('');
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [expandedSources, setExpandedSources] = useState({});
  const [expandedStages, setExpandedStages] = useState({});
  const chatEndRef = useRef(null);

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  useEffect(() => {
    if (username.trim() && email.trim()) {
      fetchHistory();
    }
  }, [username, email]);

  const fetchHistory = async () => {
    if (!username.trim() || !email.trim()) return;
    const userId = `${username.trim().toLowerCase()}::${email
      .trim()
      .toLowerCase()}`;
    try {
      const response = await fetch(
        `${API_BASE_URL}/history?user_id=${encodeURIComponent(
          userId
        )}&username=${encodeURIComponent(
          username.trim()
        )}&email=${encodeURIComponent(email.trim())}`
      );
      const data = await response.json();
      if (data.success) {
        setChatHistory(data.data);
      }
    } catch (error) {
      console.error('Error fetching history:', error);
    }
  };

  const handleAsk = async () => {
    if (!question.trim() || loading) return;
    if (!username.trim() || !email.trim()) {
      alert('Please enter username and email to continue.');
      return;
    }
    const userId = `${username.trim().toLowerCase()}::${email
      .trim()
      .toLowerCase()}`;

    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: question.trim(),
          session_id: SESSION_ID,
          user_id: userId,
          username: username.trim(),
          email: email.trim(),
        }),
      });

      const data = await response.json();
      if (data.success) {
        setChatHistory((prev) => [data.data, ...prev]);
        setQuestion('');
      } else {
        alert('Error: ' + data.error);
      }
    } catch (error) {
      console.error('Error asking question:', error);
      alert('Failed to get answer. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleFollowUp = async () => {
    if (!question.trim() || loading || chatHistory.length === 0) return;
    if (!username.trim() || !email.trim()) {
      alert('Please enter username and email to continue.');
      return;
    }
    const userId = `${username.trim().toLowerCase()}::${email
      .trim()
      .toLowerCase()}`;

    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/followup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: question.trim(),
          session_id: SESSION_ID,
          user_id: userId,
          username: username.trim(),
          email: email.trim(),
        }),
      });

      const data = await response.json();
      if (data.success) {
        setChatHistory((prev) => [data.data, ...prev]);
        setQuestion('');
      } else {
        alert('Error: ' + data.error);
      }
    } catch (error) {
      console.error('Error with follow-up:', error);
      alert('Failed to get answer. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleTranslate = async () => {
    if (chatHistory.length === 0 || loading) return;
    if (!username.trim() || !email.trim()) {
      alert('Please enter username and email to continue.');
      return;
    }
    const userId = `${username.trim().toLowerCase()}::${email
      .trim()
      .toLowerCase()}`;

    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/translate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: SESSION_ID,
          user_id: userId,
          username: username.trim(),
          email: email.trim(),
        }),
      });

      const data = await response.json();
      if (data.success) {
        setChatHistory((prev) => [data.data, ...prev]);
      } else {
        alert(data.error || 'Translation failed');
      }
    } catch (error) {
      console.error('Error translating:', error);
      alert('Failed to translate. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = async () => {
    if (!window.confirm('Clear all chat history?')) return;
    if (!username.trim() || !email.trim()) {
      alert('Please enter username and email to continue.');
      return;
    }
    const userId = `${username.trim().toLowerCase()}::${email
      .trim()
      .toLowerCase()}`;

    try {
      const response = await fetch(`${API_BASE_URL}/clear`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: SESSION_ID,
          user_id: userId,
          username: username.trim(),
          email: email.trim(),
        }),
      });

      const data = await response.json();
      if (data.success) {
        setChatHistory([]);
      }
    } catch (error) {
      console.error('Error clearing history:', error);
    }
  };

  const handleExportPDF = async () => {
    if (chatHistory.length === 0) return;
    if (!username.trim() || !email.trim()) {
      alert('Please enter username and email to continue.');
      return;
    }
    const userId = `${username.trim().toLowerCase()}::${email
      .trim()
      .toLowerCase()}`;

    try {
      const response = await fetch(`${API_BASE_URL}/export/pdf`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: SESSION_ID,
          user_id: userId,
          username: username.trim(),
          email: email.trim(),
        }),
      });

      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'ifrs_chat_history.pdf';
        a.click();
        window.URL.revokeObjectURL(url);
      } else {
        alert('PDF export failed');
      }
    } catch (error) {
      console.error('Error exporting PDF:', error);
    }
  };

  const handleExportHTML = async () => {
    if (chatHistory.length === 0) return;
    if (!username.trim() || !email.trim()) {
      alert('Please enter username and email to continue.');
      return;
    }
    const userId = `${username.trim().toLowerCase()}::${email
      .trim()
      .toLowerCase()}`;

    try {
      const response = await fetch(`${API_BASE_URL}/export/html`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: SESSION_ID,
          user_id: userId,
          username: username.trim(),
          email: email.trim(),
        }),
      });

      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'ifrs_chat_history.html';
        a.click();
        window.URL.revokeObjectURL(url);
      } else {
        alert('HTML export failed');
      }
    } catch (error) {
      console.error('Error exporting HTML:', error);
    }
  };

  const toggleSources = (index) => {
    setExpandedSources((prev) => ({
      ...prev,
      [index]: !prev[index],
    }));
  };

  const toggleStages = (index) => {
    setExpandedStages((prev) => ({
      ...prev,
      [index]: !prev[index],
    }));
  };

  const formatDuration = (seconds) => {
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}m ${secs}s`;
  };

  const downloadTableCSV = (csvData, tableIndex) => {
    // Create blob from CSV data
    const blob = new Blob([csvData], { type: 'text/csv;charset=utf-8;' });
    const url = window.URL.createObjectURL(blob);

    // Create temporary download link
    const link = document.createElement('a');
    link.href = url;
    link.download = `answer_table_${tableIndex}.csv`;

    // Trigger download
    document.body.appendChild(link);
    link.click();

    // Cleanup
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="header-title">
            <Book className="header-icon" size={32} />
            <h1>IFRS Digital Co-worker</h1>
          </div>
          <div className="header-subtitle">
            Powered by AI • Retrieval-Augmented Generation
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="main-container">
        {/* Sidebar */}
        <aside className="sidebar">
          <div className="sidebar-section">
            <h3 className="sidebar-title">User</h3>
            <div className="user-fields">
              <input
                type="text"
                className="user-input"
                placeholder="Username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                disabled={loading}
              />
              <input
                type="email"
                className="user-input"
                placeholder="Email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                disabled={loading}
              />
            </div>
          </div>
          <div className="sidebar-section">
            <h3 className="sidebar-title">Recent Queries</h3>
            <div className="recent-queries">
              {chatHistory.slice(0, 5).map((chat, idx) => (
                <div key={idx} className="recent-query-item">
                  <MessageSquare size={14} />
                  <span>{chat.question.substring(0, 50)}...</span>
                </div>
              ))}
              {chatHistory.length === 0 && (
                <p className="empty-state">No queries yet</p>
              )}
            </div>
          </div>
        </aside>

        {/* Chat Area */}
        <main className="chat-container">
          {/* Input Section */}
          <div className="input-section">
            <div className="input-wrapper">
              <input
                type="text"
                className="question-input"
                placeholder="Ask your IFRS-related question..."
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleAsk()}
                disabled={loading}
              />
              {loading && (
                <Loader2 className="input-loader" size={20} />
              )}
            </div>

            <div className="action-buttons">
              <motion.button
                className="btn btn-primary"
                onClick={handleAsk}
                disabled={loading || !question.trim()}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <Send size={18} />
                Ask
              </motion.button>

              <motion.button
                className="btn btn-secondary"
                onClick={handleFollowUp}
                disabled={loading || !question.trim() || chatHistory.length === 0}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <Sparkles size={18} />
                Follow-up
              </motion.button>

              <motion.button
                className="btn btn-secondary"
                onClick={handleTranslate}
                disabled={loading || chatHistory.length === 0}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <Languages size={18} />
                Translate to Arabic
              </motion.button>

              <motion.button
                className="btn btn-secondary"
                onClick={handleExportPDF}
                disabled={chatHistory.length === 0}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <Download size={18} />
                Export PDF
              </motion.button>

              <motion.button
                className="btn btn-secondary"
                onClick={handleExportHTML}
                disabled={chatHistory.length === 0}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <Download size={18} />
                Export HTML
              </motion.button>

              <motion.button
                className="btn btn-danger"
                onClick={handleClear}
                disabled={chatHistory.length === 0}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <Trash2 size={18} />
                Clear Chat
              </motion.button>
            </div>
          </div>

          {/* Chat History */}
          <div className="chat-history">
            {chatHistory.length === 0 && !loading && (
              <div className="empty-chat-state">
                <Book size={64} className="empty-icon" />
                <h2>Welcome to IFRS Digital Co-worker</h2>
                <p>Ask any IFRS-related question to get started</p>
              </div>
            )}

            <AnimatePresence>
              {chatHistory.map((chat, index) => (
                <motion.div
                  key={index}
                  className="chat-message"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <div className="message-header">
                    <span className="message-badge">
                      Answer from Database • {chat.kb}
                    </span>
                    {chat.time_taken_sec && (
                      <span className="message-time">
                        <Clock size={14} />
                        {formatDuration(chat.time_taken_sec)}
                      </span>
                    )}
                  </div>

                  {/* Question */}
                  <div
                    className={`message-question ${
                      chat.is_arabic ? 'arabic-text' : ''
                    }`}
                  >
                    <strong>👤 User:</strong>
                    {chat.is_arabic ? (
                      <div className="markdown-content">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {chat.question}
                        </ReactMarkdown>
                      </div>
                    ) : (
                      <span> {chat.question}</span>
                    )}
                  </div>

                  {/* Answer */}
                  <div
                    className={`message-answer ${
                      chat.is_arabic ? 'arabic-text' : ''
                    }`}
                  >
                    <strong>🤖 Assistant:</strong>
                    {chat.is_arabic ? (
                      <div className="markdown-content">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {chat.answer}
                        </ReactMarkdown>
                      </div>
                    ) : (
                      <div className="markdown-content">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {chat.answer}
                        </ReactMarkdown>
                      </div>
                    )}
                  </div>

                  {/* Tables */}
                  {chat.tables && chat.tables.length > 0 && (
                    <div className="answer-tables">
                      <div className="answer-tables-label">Tables</div>
                      {chat.tables.map((table, tIdx) => (
                        <div key={tIdx} className="answer-table">
                          {table.table_name && (
                            <div className="answer-table-title">
                              {table.table_name}
                            </div>
                          )}
                          <div className="answer-table-wrap">
                            <table className="answer-table-grid">
                              <thead>
                                <tr>
                                  {(table.columns || []).map((col, cIdx) => (
                                    <th key={cIdx}>{col}</th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {(table.rows || []).map((row, rIdx) => (
                                  <tr key={rIdx}>
                                    {(row || []).map((cell, cIdx) => (
                                      <td key={cIdx}>
                                        {cell === null ||
                                        cell === undefined ||
                                        String(cell).trim() === ''
                                          ? '—'
                                          : String(cell)}
                                      </td>
                                    ))}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Table Downloads */}
                  {chat.table_data && chat.table_data.length > 0 && (
                    <div className="table-downloads">
                      <div
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '8px',
                          marginBottom: '12px',
                          fontSize: '14px',
                          color: '#666',
                        }}
                      >
                        <FileText size={16} />
                        <span>
                          {chat.table_data.length === 1
                            ? '1 table detected in answer'
                            : `${chat.table_data.length} tables detected in answer`}
                        </span>
                      </div>
                      <div
                        style={{
                          display: 'flex',
                          flexWrap: 'wrap',
                          gap: '8px',
                        }}
                      >
                        {chat.table_data.map((table) => (
                          <button
                            key={table.index}
                            className="download-table-btn"
                            onClick={() =>
                              downloadTableCSV(table.csv, table.index)
                            }
                            style={{
                              display: 'flex',
                              alignItems: 'center',
                              gap: '6px',
                              padding: '8px 12px',
                              backgroundColor: '#f0f0f0',
                              border: '1px solid #ddd',
                              borderRadius: '6px',
                              cursor: 'pointer',
                              fontSize: '13px',
                              transition: 'all 0.2s',
                            }}
                            onMouseOver={(e) => {
                              e.currentTarget.style.backgroundColor = '#e0e0e0';
                              e.currentTarget.style.borderColor = '#bbb';
                            }}
                            onMouseOut={(e) => {
                              e.currentTarget.style.backgroundColor = '#f0f0f0';
                              e.currentTarget.style.borderColor = '#ddd';
                            }}
                          >
                            <Download size={14} />
                            Download Table {table.index} as CSV
                            <span
                              style={{
                                fontSize: '11px',
                                color: '#999',
                                marginLeft: '4px',
                              }}
                            >
                              ({table.row_count} rows × {table.col_count} cols)
                            </span>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Stage Answers */}
                  {chat.stage_answers &&
                    Object.keys(chat.stage_answers).length > 0 && (
                      <div className="stage-answers">
                        <button
                          className="collapse-button"
                          onClick={() => toggleStages(index)}
                        >
                          {expandedStages[index] ? (
                            <ChevronUp size={16} />
                          ) : (
                            <ChevronDown size={16} />
                          )}
                          Intermediate Answers
                        </button>

                        {expandedStages[index] && (
                          <div className="stage-content">
                            {Object.entries(chat.stage_answers).map(
                              ([stage, content]) =>
                                content && (
                                  <div key={stage} className="stage-item">
                                    <h4>{stage}</h4>
                                    <div className="markdown-content">
                                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                        {content}
                                      </ReactMarkdown>
                                    </div>
                                  </div>
                                )
                            )}
                          </div>
                        )}
                      </div>
                    )}

                  {/* References */}
                  {chat.sources && chat.sources.length > 0 && (
                    <div className="references">
                      <button
                        className="collapse-button"
                        onClick={() => toggleSources(index)}
                      >
                        {expandedSources[index] ? (
                          <ChevronUp size={16} />
                        ) : (
                          <ChevronDown size={16} />
                        )}
                        📚 References ({chat.sources.length})
                      </button>

                      {expandedSources[index] && (
                        <div className="references-content">
                          {chat.sources.map((source, srcIdx) => (
                            <div key={srcIdx} className="reference-item">
                              <div className="reference-meta">
                                <div className="meta-item">
                                  <strong>Document:</strong> {source.doc_name}
                                </div>
                                <div className="meta-item">
                                  <strong>Chapter:</strong> {source.chapter_name}
                                </div>
                                <div className="meta-item">
                                  <strong>Paragraph:</strong> {source.para_number}
                                </div>
                                {source.header !== '—' && (
                                  <div className="meta-item">
                                    <strong>Header:</strong> {source.header}
                                  </div>
                                )}
                                <div className="meta-item">
                                  <strong>Page:</strong> {source.page}
                                </div>
                                <div className="meta-item">
                                  <strong>Publisher:</strong> {source.publisher}
                                </div>
                              </div>
                              <div className="reference-excerpt">
                                <strong>Excerpt:</strong>
                                <p>{source.excerpt}</p>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </motion.div>
              ))}
            </AnimatePresence>

            <div ref={chatEndRef} />
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
