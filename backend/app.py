"""
Flask Backend API for IFRS Digital Co-worker
Reuses logic from IFRS_chat_streamlit_final.py
"""

import os
import sys

# Force unbuffered output for real-time logging
os.environ['PYTHONUNBUFFERED'] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from io import BytesIO
import traceback
import pandas as pd

from session_store import SessionStore

# Add parent directory to path to import from RAG engine
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions from the RAG engine (no Streamlit dependencies)
from rag_engine.answer import answer_with_refine_chain
from rag_engine.exports import (
    REPORTLAB_AVAILABLE,
    FPDF_AVAILABLE,
    _build_pdf_reportlab,
    _build_pdf_fpdf,
    _build_html_export,
)
from rag_engine.formatting import (
    _format_duration,
    _make_excerpt,
    _unify_metadata,
    fix_citation_format,
    format_visible_answer,
    remove_citations,
    sanitize_text,
)
from rag_engine.tables import extract_markdown_tables_as_dfs
from rag_engine.translate import translate_to_arabic
from rag_engine.engine import filter_page_zero_references

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

session_store = SessionStore()


def _get_user_context(data, args):
    username = (data.get("username") if data else None) or args.get("username")
    email = (data.get("email") if data else None) or args.get("email")
    user_id = (data.get("user_id") if data else None) or args.get("user_id")
    if not username or not email:
        return None, None, None, "username and email are required"
    if not user_id:
        user_id = f"{username.strip().lower()}::{email.strip().lower()}"
    return user_id, username.strip(), email.strip(), None


def get_session(user_id):
    history = session_store.load_history(user_id)
    return {"chat_history": history}


def save_session(user_id, session):
    session_store.save_history(user_id, session.get("chat_history", []))


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'IFRS Backend API Running'})


@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Main endpoint to ask a question"""
    try:
        data = request.json
        question = data.get('question', '').strip()
        user_id, username, email, err = _get_user_context(data, request.args)
        if err:
            return jsonify({'error': err}), 400

        print(f"\n{'='*80}")
        print(f"NEW QUESTION RECEIVED: {question}")
        print(f"User ID: {user_id}")
        print(f"{'='*80}\n")
        sys.stdout.flush()

        if not question:
            return jsonify({'error': 'Question is required'}), 400

        session = get_session(user_id)

        # Call the main RAG function
        import time
        start_t = time.perf_counter()
        res = answer_with_refine_chain(question)
        elapsed = time.perf_counter() - start_t

        # Get raw answer and exception section
        answer = res.get("answer_text") or res.get("answer") or ""
        exception_section = res.get("exception_section", "")

        print(f"\n{'='*80}")
        print(f"FLASK BACKEND - COMBINING SECTIONS FOR REACT FRONTEND")
        print(f"{'='*80}")
        print(f"RAW MAIN ANSWER ({len(answer)} chars):")
        print(f"{'-'*80}")
        print(answer)
        print(f"{'-'*80}\n")

        # Combine raw sections BEFORE formatting
        if exception_section:
            print(f"RAW EXCEPTION SECTION ({len(exception_section)} chars):")
            print(f"{'-'*80}")
            print(exception_section)
            print(f"{'-'*80}\n")

            answer = answer.rstrip() + "\n\n" + exception_section

            print(f"COMBINED RAW ANSWER (before formatting, {len(answer)} chars):")
            print(f"{'-'*80}")
            print(answer)
            print(f"{'-'*80}\n")
        else:
            print(f"Exception section is EMPTY, using main answer only\n")

        # Format the complete combined answer
        if answer.strip().lower() != "sources not found.":
            answer = format_visible_answer(answer)

            print(f"FINAL FORMATTED ANSWER FOR REACT FRONTEND ({len(answer)} chars):")
            print(f"{'-'*80}")
            print(answer)
            print(f"{'-'*80}")
            print(f"{'='*80}\n")

        # Process stage answers
        stage_clean = {}
        for k, v in (res.get("stage_answers") or {}).items():
            vv = v if isinstance(v, str) else ""
            vv = format_visible_answer(vv)
            vv = remove_citations(vv)
            stage_clean[k] = vv

        # Filter sources
        display_sources = filter_page_zero_references(res["sources"])

        # Handle case where no sources are returned
        if display_sources is None:
            display_sources = []

        # Convert sources to serializable format
        sources_list = []
        for doc in display_sources:
            meta = _unify_metadata(getattr(doc, "metadata", {}) or {})
            chunk_text = getattr(doc, "page_content", "") or ""
            sources_list.append({
                'doc_name': meta.get('doc_name', 'Document'),
                'chapter': meta.get('chapter', '—'),
                'chapter_name': meta.get('chapter_name', '—'),
                'para_number': meta.get('para_number', '—'),
                'header': meta.get('header', '—'),
                'page': meta.get('page', 0),
                'publisher': meta.get('publisher', '—'),
                'excerpt': _make_excerpt(chunk_text, max_chars=900)
            })

        # Tables from model JSON payload
        tables_payload = res.get("tables") or []
        table_data = []
        for idx, t in enumerate(tables_payload, start=1):
            cols = t.get("columns") or []
            rows = t.get("rows") or []
            try:
                df = pd.DataFrame(rows, columns=cols)
            except Exception:
                continue
            df = df.fillna("").applymap(lambda v: "—" if str(v).strip() == "" else v)
            table_data.append({
                'index': idx,
                'csv': df.to_csv(index=False),
                'row_count': len(df),
                'col_count': len(df.columns)
            })
        has_tables = len(table_data) > 0

        # Create chat entry
        chat_entry = {
            "mode": "Answer from Database",
            "kb": "IFRS A/B/C",
            "user_id": user_id,
            "username": username,
            "email": email,
            "question": question,
            "answer": answer,
            "stage_answers": stage_clean,
            "sources": sources_list,
            "time_taken_sec": elapsed,
            "has_tables": has_tables,
            "table_data": table_data,
            "tables": tables_payload,
            "is_arabic": False
        }

        # Store in session history
        session['chat_history'].insert(0, chat_entry)
        save_session(user_id, session)

        print(f"\n{'='*80}")
        print(f"ANSWER COMPLETED in {elapsed:.2f}s")
        print(f"Sources found: {len(sources_list)}")
        print(f"Tables detected: {len(table_data)}")
        print(f"{'='*80}\n")

        print(f"{'='*80}")
        print(f"JSON BEING SENT TO REACT FRONTEND")
        print(f"{'='*80}")
        print(f"Structure: {{success: True, data: chat_entry}}")
        print(f"chat_entry['answer'] length: {len(chat_entry['answer'])} chars (COMBINED main + exception)")
        print(f"chat_entry['sources']: {len(chat_entry['sources'])} sources")
        print(f"chat_entry['stage_answers']: {len(chat_entry['stage_answers'])} stages")
        print(f"chat_entry['has_tables']: {chat_entry['has_tables']}")
        print(f"{'='*80}\n")
        sys.stdout.flush()

        return jsonify({
            'success': True,
            'data': chat_entry
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/followup', methods=['POST'])
def followup_question():
    """Follow-up question endpoint"""
    try:
        data = request.json
        question = data.get('question', '').strip()
        user_id, username, email, err = _get_user_context(data, request.args)
        if err:
            return jsonify({'error': err}), 400

        if not question:
            return jsonify({'error': 'Question is required'}), 400

        session = get_session(user_id)

        # Combine with last non-Arabic question (ignore translated entries)
        prev_q = ''
        for entry in session.get('chat_history', []):
            if not entry.get('is_arabic'):
                prev_q = entry.get('question', '')
                break
        combined = f"{prev_q} {question}".strip()

        # Call ask endpoint logic
        import time
        start_t = time.perf_counter()
        res = answer_with_refine_chain(combined)
        elapsed = time.perf_counter() - start_t

        # Get raw answer and exception section (same as ask endpoint)
        answer = res.get("answer_text") or res.get("answer") or ""
        exception_section = res.get("exception_section", "")

        # Combine raw sections BEFORE formatting
        if exception_section:
            answer = answer.rstrip() + "\n\n" + exception_section

        # Format the complete combined answer
        if answer.strip().lower() != "sources not found.":
            answer = format_visible_answer(answer)

        stage_clean = {}
        for k, v in (res.get("stage_answers") or {}).items():
            vv = v if isinstance(v, str) else ""
            vv = format_visible_answer(vv)
            vv = remove_citations(vv)
            stage_clean[k] = vv

        display_sources = filter_page_zero_references(res["sources"])

        sources_list = []
        for doc in display_sources:
            meta = _unify_metadata(getattr(doc, "metadata", {}) or {})
            chunk_text = getattr(doc, "page_content", "") or ""
            sources_list.append({
                'doc_name': meta.get('doc_name', 'Document'),
                'chapter': meta.get('chapter', '—'),
                'chapter_name': meta.get('chapter_name', '—'),
                'para_number': meta.get('para_number', '—'),
                'header': meta.get('header', '—'),
                'page': meta.get('page', 0),
                'publisher': meta.get('publisher', '—'),
                'excerpt': _make_excerpt(chunk_text, max_chars=900)
            })

        tables_payload = res.get("tables") or []
        table_data = []
        for idx, t in enumerate(tables_payload, start=1):
            cols = t.get("columns") or []
            rows = t.get("rows") or []
            try:
                df = pd.DataFrame(rows, columns=cols)
            except Exception:
                continue
            df = df.fillna("").applymap(lambda v: "—" if str(v).strip() == "" else v)
            table_data.append({
                'index': idx,
                'csv': df.to_csv(index=False),
                'row_count': len(df),
                'col_count': len(df.columns)
            })
        has_tables = len(table_data) > 0

        chat_entry = {
            "mode": "Answer from Database",
            "kb": "IFRS A/B/C",
            "user_id": user_id,
            "username": username,
            "email": email,
            "question": combined,
            "answer": answer,
            "stage_answers": stage_clean,
            "sources": sources_list,
            "time_taken_sec": elapsed,
            "has_tables": has_tables,
            "table_data": table_data,
            "tables": tables_payload,
            "is_arabic": False
        }

        session['chat_history'].insert(0, chat_entry)
        save_session(user_id, session)

        return jsonify({
            'success': True,
            'data': chat_entry
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/translate', methods=['POST'])
def translate():
    """Translate latest answer to Arabic"""
    try:
        data = request.json
        user_id, username, email, err = _get_user_context(data, request.args)
        if err:
            return jsonify({'error': err}), 400

        session = get_session(user_id)

        if not session['chat_history']:
            return jsonify({'error': 'No chat history to translate'}), 400

        latest = session['chat_history'][0]

        if latest.get('is_arabic'):
            return jsonify({'error': 'Already translated to Arabic'}), 400

        # Translate
        q_ar = translate_to_arabic(latest["question"])
        a_ar = translate_to_arabic(latest["answer"])

        q_ar = remove_citations(fix_citation_format(q_ar))
        a_ar = remove_citations(fix_citation_format(a_ar))

        # Create Arabic entry
        arabic_entry = {
            "mode": latest.get("mode", "Answer from Database"),
            "kb": latest.get("kb", "IFRS A/B/C"),
            "user_id": user_id,
            "username": username,
            "email": email,
            "question": q_ar,
            "answer": a_ar,
            "sources": latest.get("sources", []),
            "stage_answers": latest.get("stage_answers", {}),
            "tables": latest.get("tables", []),
            "table_data": latest.get("table_data", []),
            "has_tables": latest.get("has_tables", False),
            "is_arabic": True,
            "time_taken_sec": latest.get("time_taken_sec"),
        }

        session['chat_history'].insert(0, arabic_entry)
        save_session(user_id, session)

        return jsonify({
            'success': True,
            'data': arabic_entry
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get chat history"""
    user_id, username, email, err = _get_user_context({}, request.args)
    if err:
        return jsonify({'error': err}), 400
    session = get_session(user_id)

    return jsonify({
        'success': True,
        'data': session['chat_history']
    })


@app.route('/api/clear', methods=['POST'])
def clear_history():
    """Clear chat history"""
    data = request.json
    user_id, username, email, err = _get_user_context(data, request.args)
    if err:
        return jsonify({'error': err}), 400

    session = get_session(user_id)
    session['chat_history'] = []
    save_session(user_id, session)

    return jsonify({
        'success': True,
        'message': 'Chat history cleared'
    })


@app.route('/api/export/csv', methods=['POST'])
def export_csv():
    """Export table as CSV (supports multiple tables by index)"""
    try:
        data = request.json
        table_markdown = data.get('table_markdown', '')
        table_index = data.get('table_index', 0)  # 0-based index, default to first table

        if not table_markdown:
            return jsonify({'error': 'No table data provided'}), 400

        # Extract table
        tables = extract_markdown_tables_as_dfs(table_markdown)

        if not tables:
            return jsonify({'error': 'No valid table found'}), 400

        # Validate table index
        if table_index < 0 or table_index >= len(tables):
            return jsonify({'error': f'Invalid table index. Found {len(tables)} table(s).'}), 400

        # Convert specified table to CSV
        df = tables[table_index]
        csv_buffer = BytesIO()
        csv_buffer.write(df.to_csv(index=False).encode('utf-8'))
        csv_buffer.seek(0)

        return send_file(
            csv_buffer,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'table_{table_index + 1}_export.csv'
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/export/pdf', methods=['POST'])
def export_pdf():
    """Export conversation as PDF"""
    try:
        data = request.json
        user_id, username, email, err = _get_user_context(data, request.args)
        if err:
            return jsonify({'error': err}), 400

        session = get_session(user_id)

        if not session['chat_history']:
            return jsonify({'error': 'No chat history to export'}), 400

        # Convert sources back to Document-like objects
        history_for_pdf = []
        for chat in session['chat_history']:
            # Create simple objects that have the required attributes
            class SimpleDoc:
                def __init__(self, metadata, page_content):
                    self.metadata = metadata
                    self.page_content = page_content

            sources_as_docs = []
            for src in chat.get('sources', []):
                doc = SimpleDoc(
                    metadata={
                        'doc_name': src.get('doc_name'),
                        'chapter': src.get('chapter'),
                        'chapter_name': src.get('chapter_name'),
                        'para_number': src.get('para_number'),
                        'header': src.get('header'),
                        'page': src.get('page'),
                        'publisher': src.get('publisher'),
                    },
                    page_content=src.get('excerpt', '')
                )
                sources_as_docs.append(doc)

            history_for_pdf.append({
                'question': chat['question'],
                'answer': chat['answer'],
                'stage_answers': chat.get('stage_answers', {}),
                'sources': sources_as_docs,
                'time_taken_sec': chat.get('time_taken_sec', 0),
                'tables': chat.get('tables', []),
            })

        # Build PDF
        if REPORTLAB_AVAILABLE:
            pdf_bytes = _build_pdf_reportlab(history_for_pdf)
        elif FPDF_AVAILABLE:
            pdf_bytes = _build_pdf_fpdf(history_for_pdf)
        else:
            return jsonify({'error': 'No PDF library available'}), 500

        pdf_buffer = BytesIO(pdf_bytes)
        pdf_buffer.seek(0)

        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='ifrs_chat_history.pdf'
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/export/html', methods=['POST'])
def export_html():
    """Export conversation as HTML"""
    try:
        data = request.json
        user_id, username, email, err = _get_user_context(data, request.args)
        if err:
            return jsonify({'error': err}), 400

        session = get_session(user_id)

        if not session['chat_history']:
            return jsonify({'error': 'No chat history to export'}), 400

        history_for_export = []
        for chat in session['chat_history']:
            history_for_export.append({
                'question': chat['question'],
                'answer': chat['answer'],
                'time_taken_sec': chat.get('time_taken_sec', 0),
                'tables': chat.get('tables', []),
            })

        html_text = _build_html_export(history_for_export)
        html_buffer = BytesIO(html_text.encode("utf-8"))

        return send_file(
            html_buffer,
            mimetype='text/html; charset=utf-8',
            as_attachment=True,
            download_name='ifrs_chat_history.html'
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("Starting IFRS Digital Co-worker Backend...")
    print("API will be available at http://localhost:3000")
    app.run(debug=True, host='0.0.0.0', port=3000)
