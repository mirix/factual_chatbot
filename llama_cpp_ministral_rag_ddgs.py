import gradio as gr
import requests
import subprocess
import re
import os
import time
import hashlib
import signal
import sys
from ddgs import DDGS
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# ══════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════
MODEL_PATH = "./Ministral-3-14B-Instruct-2512-absolute-heresy.Q8_0.gguf"
LLAMA_SERVER_PATH = "./llama.cpp/build/bin/llama-server"

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

DEFAULT_N_CTX = 16384
DEFAULT_N_GPU_LAYERS = -1

DEFAULT_TEMP = 0.4
DEFAULT_TOP_P = 0.9
DEFAULT_MIN_P = 0.05
DEFAULT_REPEAT_PENALTY = 1.15
DEFAULT_MAX_TOKENS = 2048


AVAILABLE_ENGINES = ['bing', 'brave', 'duckduckgo', 'google', 'grokipedia', 'mojeek', 'yandex', 'yahoo', 'wikipedia']
DEFAULT_ENGINES = ['brave', 'duckduckgo','mojeek', 'yandex']
DEFAULT_MAX_SEARCH_RESULTS = 20

# ══════════════════════════════════════════════
# Server Manager (Start/Check/Stop)
# ══════════════════════════════════════════════
class ServerManager:
    def __init__(self, host, port, model_path, server_path, n_ctx, n_gpu_layers):
        self.host = host
        self.port = port
        self.model_path = model_path
        self.server_path = server_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.process = None
        self.log_file = f"llama_server_{port}.log"
        self.started = False

    def is_running(self):
        try:
            resp = requests.get(f"{SERVER_URL}/health", timeout=2)
            return resp.status_code == 200
        except:
            return False

    def start(self):
        if self.started and self.is_running():
            print(f"[SERVER] ✅ Already running on {SERVER_URL}")
            return True

        # Check paths
        if not os.path.exists(self.server_path):
            print(f"[SERVER] ❌ Server binary not found: {self.server_path}")
            print(f"[SERVER] Hint: Build llama.cpp with: cd llama.cpp && cmake -B build && cmake --build build --config Release")
            return False
        
        if not os.path.exists(self.model_path):
            print(f"[SERVER] ❌ Model not found: {self.model_path}")
            return False

        print(f"\n{'='*60}")
        print(f"[SERVER] 🚀 Starting llama.cpp server...")
        print(f"[SERVER] Binary: {self.server_path}")
        print(f"[SERVER] Model: {self.model_path}")
        print(f"[SERVER] Port: {self.port}")
        print(f"[SERVER] Log file: {self.log_file}")
        print("-" * 60)

        cmd = [
            self.server_path,
            "-m", self.model_path,
            "-c", str(self.n_ctx),
            "-ngl", str(self.n_gpu_layers),
            "--host", self.host,
            "--port", str(self.port),
            "--log-disable"
        ]

        try:
            # Open log file for server output
            log_fh = open(self.log_file, "w")
            
            self.process = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True
            )
            
            print(f"[SERVER] Process PID: {self.process.pid}")
            
            # Wait for server to be ready (max 120 seconds)
            start_time = time.time()
            last_log_line = 0
            
            for i in range(120):
                # Check if process died
                if self.process.poll() is not None:
                    print(f"[SERVER] ❌ Server process died after {i+1} seconds")
                    self._show_log_tail()
                    return False
                
                # Check if server is healthy
                if self.is_running():
                    elapsed = time.time() - start_time
                    print(f"\n[SERVER] ✅ Ready after {i+1} seconds ({elapsed:.1f}s)")
                    print("-" * 60)
                    self.started = True
                    return True
                
                # Show log progress every 5 seconds
                if i % 5 == 0:
                    self._show_log_tail(last_log_line)
                    last_log_line = self._get_log_line_count()
                    print(f"[SERVER] Waiting... ({i+1}s)")
                
                time.sleep(1)
            
            print(f"[SERVER] ❌ Failed to start within 120 seconds")
            self._show_log_tail()
            return False
            
        except Exception as e:
            print(f"[SERVER] ❌ Exception: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_log_line_count(self):
        try:
            with open(self.log_file, "r") as f:
                return sum(1 for _ in f)
        except:
            return 0

    def _show_log_tail(self, from_line=0):
        """Show recent log output"""
        try:
            with open(self.log_file, "r") as f:
                lines = f.readlines()
                if from_line > 0:
                    lines = lines[from_line:]
                else:
                    lines = lines[-10:]
                
                for line in lines:
                    line = line.strip()
                    if line:
                        print(f"[LLAMA] {line}")
        except Exception as e:
            print(f"[LLAMA] (Could not read log: {e})")

    def stop(self):
        if self.process:
            try:
                print(f"\n[SERVER] Stopping server (PID: {self.process.pid})...")
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=5)
                print("[SERVER] ✅ Stopped")
            except Exception as e:
                print(f"[SERVER] Warning: {e}")
                try:
                    self.process.kill()
                except:
                    pass

# Initialize global server manager
server_mgr = ServerManager(
    host=SERVER_HOST,
    port=SERVER_PORT,
    model_path=MODEL_PATH,
    server_path=LLAMA_SERVER_PATH,
    n_ctx=DEFAULT_N_CTX,
    n_gpu_layers=DEFAULT_N_GPU_LAYERS
)

# ══════════════════════════════════════════════
# LLM API Client (STREAMING)
# ══════════════════════════════════════════════
def query_llama_api_stream(prompt, max_tokens, temperature, top_p, min_p, repeat_penalty):
    """Stream tokens from llama.cpp server using SSE"""
    if not server_mgr.is_running():
        raise ConnectionError("LLM server is not running")

    payload = {
        "prompt": prompt,
        "n_predict": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "min_p": float(min_p),
        "repeat_penalty": float(repeat_penalty),
        "stop": ["</s>", "[INST]", "[/INST]"],
        "cache_prompt": True,
        "stream": True  # Enable streaming
    }

    try:
        print(f"[API] Sending streaming request to {SERVER_URL}/completion...")
        
        # Use stream=True for SSE
        resp = requests.post(
            f"{SERVER_URL}/completion", 
            json=payload, 
            timeout=300,
            stream=True
        )
        resp.raise_for_status()
        
        # Parse Server-Sent Events
        full_content = ""
        for line in resp.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix
                    if data.strip() == '[DONE]':
                        break
                    try:
                        import json
                        chunk = json.loads(data)
                        content = chunk.get("content", "")
                        full_content += content
                        yield content  # Yield each token chunk
                    except json.JSONDecodeError:
                        continue
        
        print(f"[API] ✅ Streaming complete: {len(full_content)} characters")
        
    except requests.exceptions.RequestException as e:
        print(f"[API] ❌ Request failed: {e}")
        raise ConnectionError(f"API request failed: {e}")

def query_llama_api(prompt, max_tokens, temperature, top_p, min_p, repeat_penalty):
    """Non-streaming version (for fallback)"""
    if not server_mgr.is_running():
        raise ConnectionError("LLM server is not running")

    payload = {
        "prompt": prompt,
        "n_predict": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "min_p": float(min_p),
        "repeat_penalty": float(repeat_penalty),
        "stop": ["</s>", "[INST]", "[/INST]"],
        "cache_prompt": True
    }

    try:
        resp = requests.post(f"{SERVER_URL}/completion", json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        return data.get("content", "").strip()
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"API request failed: {e}")

# ══════════════════════════════════════════════
# Search Logic
# ══════════════════════════════════════════════
def deduplicate_results(results):
    seen = set()
    unique = []
    for r in results:
        url = r.get("href", "") or r.get("url", "")
        if not url:
            continue
        url_hash = hashlib.md5(url.rstrip("/").lower().encode()).hexdigest()
        if url_hash not in seen:
            seen.add(url_hash)
            unique.append({
                "title": r.get("title", "No Title"),
                "url": url,
                "snippet": r.get("body", "") or r.get("snippet", ""),
                "engine": r.get("source", "Web")
            })
    return unique

def search_engine(engine, query, limit):
    try:
        ddgs = DDGS()
        results = []
        for r in ddgs.text(query, safesearch="off", backend=engine, max_results=limit):
            r["source"] = engine.capitalize()
            results.append(r)
        return results
    except Exception as e:
        print(f"[!] Error {engine}: {e}")
        return []

def search_ddgs(query, engines, max_results_total):
    if not engines or not query.strip():
        return []
    query = query.strip().replace("\n", " ")
    limit_per_engine = max(3, int(max_results_total / len(engines)) + 1)
    all_results = []
    with ThreadPoolExecutor(max_workers=len(engines)) as executor:
        futures = [executor.submit(search_engine, engine, query, limit_per_engine) for engine in engines]
        for future in futures:
            all_results.extend(future.result())
    unique = deduplicate_results(all_results)
    return unique[:max_results_total]

# ══════════════════════════════════════════════
# Citation Processing (Markdown)
# ══════════════════════════════════════════════
def append_sources(text: str, search_results: list) -> str:
    if not search_results:
        return text + "\n\n---\n**📚 References:**\n*No search results were returned.*\n"

    def replace_inline(match):
        try:
            n = int(match.group(1))
            if 1 <= n <= len(search_results):
                url = search_results[n - 1]["url"]
                return f"[Source {n}]({url})"
        except Exception:
            pass
        return match.group(0)

    formatted_text = re.sub(r"\[Source\s*(\d+)\]", replace_inline, text, flags=re.IGNORECASE)

    sources_block = "\n\n---\n**📚 References:**\n"
    for i, r in enumerate(search_results):
        title = r.get("title", f"Source {i+1}").strip().replace("[", "(").replace("]", ")")
        url = r.get("url", "#")
        sources_block += f"{i+1}. [{title}]({url})\n"

    return formatted_text + sources_block

# ══════════════════════════════════════════════
# Prompt Building
# ══════════════════════════════════════════════
def build_system_prompt(search_results):
    if not search_results:
        return (
            "You are a precise research assistant. "
            "Answer factually and concisely based on your knowledge. "
            "Prioritise empirical evidence and established science."
        )

    source_block = ""
    for i, r in enumerate(search_results):
        source_block += (
            f"[Source {i+1}] {r['title']}\n"
            f"URL: {r['url']}\n"
            f"Excerpt: {r['snippet']}\n\n"
        )

    return (
        "You are a precise research assistant tasked with synthesising ALL provided search results.\n\n"
        "MANDATORY REQUIREMENTS:\n"
        "1. CONSIDER EVERY SOURCE: Review all numbered search results before answering. Do not ignore, dismiss, or exclude any source based on perceived credibility, political alignment, or ideological content.\n"
        "2. CITE COMPREHENSIVELY: Every factual claim must be supported by at least one inline citation using the exact format [Source N]. Draw from multiple sources where applicable.\n"
        "3. SYNTHESISE DIVERGENT VIEWS: Where sources disagree, present the full range of evidence. Explain disagreements neutrally without privileging one perspective over another.\n"
        "4. ACKNOWLEDGE LIMITATIONS: If certain sources contain methodological weaknesses, note them objectively rather than excluding the source entirely.\n"
        "5. NO EXTERNAL KNOWLEDGE: Base your answer exclusively on the provided search results. Do not draw on prior knowledge or make claims unsupported by the sources.\n"
        "6. TRANSPARENCY: If the search results contain insufficient information to answer the question, state this explicitly rather than speculating.\n\n"
        "PROHIBITED BEHAVIOURS:\n"
        "- Dismissing sources as 'unreliable', 'biased', or 'non-mainstream' without objective methodological critique\n"
        "- Cherry-picking only sources that confirm a particular narrative\n"
        "- Making definitive claims without citation support\n"
        "- Ignoring sources that present alternative viewpoints\n\n"
        "OUTPUT FORMAT:\n"
        "- Begin with a direct answer to the question\n"
        "- Support each claim with inline citations (e.g., [Source 1], [Source 2, 3])\n"
        "- Conclude with a summary that reflects the full range of evidence\n"
        f"SEARCH RESULTS:\n{source_block}"
    )

def build_mistral_prompt(messages):
    prompt = ""
    system_content = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            system_content = content
        elif role == "user":
            if system_content:
                prompt += f"[INST] {system_content}\n\n{content} [/INST]"
                system_content = ""
            else:
                prompt += f"[INST] {content} [/INST]"
        elif role == "assistant":
            prompt += f" {content}</s>"
    return prompt

# ══════════════════════════════════════════════
# Main Chat Function (STREAMING)
# ══════════════════════════════════════════════
def chat_with_model(
    message, history,
    search_enabled, selected_engines,
    temperature, repeat_penalty,
    max_answer_tokens, max_search_results
):
    print(f"\n{'='*60}")
    print(f"[CHAT] Request received at {datetime.now().strftime('%H:%M:%S')}")
    print(f"[CHAT] Message: {message[:100]}...")
    
    # Verify server is running (should already be started at launch)
    if not server_mgr.is_running():
        print("[CHAT] ⚠️ Server not running, attempting to start...")
        if not server_mgr.start():
            yield history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "**Error:** Failed to start LLM server. Check console for details."}
            ]
            return
    else:
        print("[CHAT] Server already running ✅")

    # Perform search if enabled
    search_results = []
    if search_enabled and message.strip():
        print(f"[CHAT] Running search with {selected_engines}")
        search_results = search_ddgs(message, selected_engines, int(max_search_results))
        print(f"[CHAT] Search returned {len(search_results)} results")

    # Build system prompt with search results
    system_prompt = build_system_prompt(search_results)
    
    # Build message history for prompt
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history or []:
        messages.append(msg)
    messages.append({"role": "user", "content": message})

    # Prepare history for Chatbot
    history = history or []
    prompt = build_mistral_prompt(messages)
    
    print(f"[CHAT] Prompt length: {len(prompt)} chars")

    try:
        # Stream tokens from API
        print(f"[CHAT] Starting streaming generation...")
        raw_text = ""
        token_count = 0
        
        for chunk in query_llama_api_stream(
            prompt=prompt,
            max_tokens=int(max_answer_tokens),
            temperature=temperature,
            top_p=DEFAULT_TOP_P,
            min_p=DEFAULT_MIN_P,
            repeat_penalty=repeat_penalty
        ):
            raw_text += chunk
            token_count += 1
            
            # Yield during streaming (without references yet)
            yield history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": raw_text}
            ]
        
        print(f"[CHAT] Streaming complete: {len(raw_text)} chars, {token_count} tokens")

        # Append references AFTER streaming completes
        if raw_text.strip():
            final_text = append_sources(raw_text, search_results)
            if search_enabled and not search_results:
                final_text += "\n\n---\n*(Note: Web search was enabled, but no results were returned.)*"
        else:
            final_text = "*(No response returned.)*"

        print(f"[CHAT] Final text length: {len(final_text)} chars")
        print(f"[CHAT] Request completed ✅")
        print(f"{'='*60}\n")

        # CRITICAL: Final yield with complete message (includes references)
        # This signals to Gradio that the stream has ended
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": final_text}
        ]

    except Exception as e:
        print(f"[CHAT] ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"**Error:** {str(e)}"}
        ]

# ══════════════════════════════════════════════
# Clear Chat Function
# ══════════════════════════════════════════════
def clear_chat():
    return []

# ══════════════════════════════════════════════
# Gradio UI
# ══════════════════════════════════════════════
custom_css = """
.chatbot-container {
    max-height: 580px;
    overflow-y: auto;
}

.message-wrap {
    max-width: 100%;
}

.chatbot .prose {
    max-width: 100%;
}
"""

with gr.Blocks(title="Ministral-3-14B RAG", fill_height=True) as demo:
    gr.Markdown("## Ministral-3-14B-Instruct · RAG")

    chatbot = gr.Chatbot(
        height=580,
        label="Conversation",
        autoscroll=False,
    )

    msg = gr.Textbox(
        placeholder="Ask a question…", 
        lines=2, 
        label="", 
    )

    with gr.Row():
        submit = gr.Button("Submit", variant="primary")
        clear_btn = gr.Button("Clear Chat")

    with gr.Accordion("⚙ Settings", open=False):
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Generation**")
                temp_slider = gr.Slider(0.1, 1.5, value=DEFAULT_TEMP, label="Temperature", step=0.1)
                penalty_slider = gr.Slider(1.0, 2.0, value=DEFAULT_REPEAT_PENALTY, label="Repeat Penalty", step=0.05)
                answer_tokens_slider = gr.Slider(256, 4096, value=DEFAULT_MAX_TOKENS, label="Max Answer Tokens", step=128)
            with gr.Column():
                gr.Markdown("**Search**")
                do_search = gr.Checkbox(value=True, label="Enable Web Search")
                search_count_slider = gr.Slider(1, 40, value=DEFAULT_MAX_SEARCH_RESULTS, step=1, label="Max Results")
                engines_box = gr.Dropdown(AVAILABLE_ENGINES, value=DEFAULT_ENGINES, multiselect=True, label="Engines")

    inputs = [msg, chatbot, do_search, engines_box, temp_slider, penalty_slider, answer_tokens_slider, search_count_slider]
    outputs = [chatbot]

    submit.click(chat_with_model, inputs=inputs, outputs=outputs).then(lambda: "", None, msg)
    msg.submit(chat_with_model, inputs=inputs, outputs=outputs).then(lambda: "", None, msg)
    clear_btn.click(clear_chat, inputs=None, outputs=outputs, queue=False)

# Enable queue for streaming
demo.queue(max_size=10, default_concurrency_limit=1)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Ministral-3-14B RAG - Starting...")
    print("="*60 + "\n")
    
    # ═══════════════════════════════════════════
    # START SERVER IMMEDIATELY ON LAUNCH
    # ═══════════════════════════════════════════
    print("[MAIN] Starting LLM server before launching Gradio...")
    if not server_mgr.start():
        print("[MAIN] ⚠️ Server failed to start. App will launch anyway but queries will fail.")
    print()
    
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            css=custom_css,
            show_error=True,
        )
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user")
    finally:
        server_mgr.stop()
        print("[MAIN] Shutdown complete")
