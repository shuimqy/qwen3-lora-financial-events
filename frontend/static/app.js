const $ = (selector) => document.querySelector(selector);
const form = $("#chat-form");
const prompt = $("#prompt");
const send = $("#send");
const messages = $("#messages");
const welcome = $("#welcome");
const errorBox = $("#error");
const newChat = $("#new-chat");
const state = { history: [], context: { document: null, awaiting_document: false }, busy: false };

function resizeInput() {
  prompt.style.height = "auto";
  prompt.style.height = `${Math.min(prompt.scrollHeight, 210)}px`;
  send.disabled = state.busy || !prompt.value.trim();
}

function setBusy(value) {
  state.busy = value;
  prompt.readOnly = value;
  newChat.disabled = value;
  form.setAttribute("aria-busy", String(value));
  resizeInput();
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = !message;
}

function textElement(tag, className, text) {
  const element = document.createElement(tag);
  element.className = className;
  element.textContent = text;
  return element;
}

function appendMessage(role, content, pending = false) {
  welcome.hidden = true;
  const article = document.createElement("article");
  article.className = `message ${role}${pending ? " pending" : ""}`;
  article.append(textElement("div", "message-label", role === "user" ? "你" : "金融助手"));
  article.append(textElement("div", "message-content", content));
  messages.append(article);
  return article;
}

function renderReply(reply, target) {
  target.classList.remove("pending");
  const content = target.querySelector(".message-content");
  const extraction = reply.extraction;
  if (reply.status === "ok" && extraction?.events?.length) {
    content.textContent = `提取到 ${extraction.events.length} 个事件：`;
    extraction.events.forEach((event) => {
      const card = document.createElement("section");
      card.className = "event-card";
      card.append(textElement("h2", "event-heading", event.event_type));
      const fields = document.createElement("dl");
      fields.className = "event-fields";
      Object.entries(event.arguments).forEach(([name, values]) => {
        const field = document.createElement("div");
        field.className = "event-field";
        field.append(textElement("dt", "", name));
        field.append(textElement("dd", "", values.length ? values.join("、") : "未提取到"));
        fields.append(field);
      });
      card.append(fields);
      target.append(card);
    });
    if (extraction.warnings?.length) {
      target.append(textElement("p", "notice", "部分字段需要与原文核对。"));
    }
  } else {
    content.textContent = reply.message.content;
  }
  if (reply.status === "length_limit" && reply.route === "base_chat") {
    target.append(textElement("p", "notice", "回复达到长度上限，可能尚未结束。"));
  }
}

function scrollToLatest() {
  window.scrollTo({ top: document.body.scrollHeight, behavior: "instant" });
}

async function refreshHealth() {
  let ready = false;
  try {
    const response = await fetch("/api/health/ready", { signal: AbortSignal.timeout(6000) });
    ready = response.ok && (await response.json()).ready === true;
  } catch { /* The status indicator is sufficient for an offline service. */ }
  $("#service-status").dataset.ready = String(ready);
  $("#status-text").textContent = ready ? "可用" : "未连接";
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = prompt.value.trim();
  if (state.busy || !text) return;
  const history = [...state.history, { role: "user", content: text }];
  if (history.length > 24 || history.reduce((sum, item) => sum + item.content.length, 0) > 24000) {
    showError("对话较长，请新建对话后继续。");
    return;
  }
  showError("");
  setBusy(true);
  prompt.value = "";
  resizeInput();
  const userMessage = appendMessage("user", text);
  const pending = appendMessage("assistant", "正在回复", true);
  scrollToLatest();
  try {
    const response = await fetch("/api/v1/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history, context: state.context }),
      signal: AbortSignal.timeout(240000),
    });
    const reply = await response.json();
    if (!response.ok) {
      throw new Error(reply.error?.message || "输入内容不符合要求，请缩短内容后重试。");
    }
    if (reply.message?.role !== "assistant" || typeof reply.message.content !== "string" || !reply.context) {
      throw new Error("暂时没有收到有效回复，请重试。");
    }
    renderReply(reply, pending);
    state.history = [...history, reply.message];
    state.context = reply.context;
  } catch (error) {
    pending.remove();
    userMessage.remove();
    welcome.hidden = state.history.length !== 0;
    prompt.value = text;
    showError(error.name === "TimeoutError" ? "回复等待超时，请稍后重试。" :
      error instanceof TypeError ? "暂时无法连接服务，请稍后重试。" : error.message);
  } finally {
    setBusy(false);
    prompt.focus({ preventScroll: true });
    scrollToLatest();
    refreshHealth();
  }
});

prompt.addEventListener("input", resizeInput);
prompt.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
    event.preventDefault();
    if (!send.disabled) form.requestSubmit();
  }
});

newChat.addEventListener("click", () => {
  if (state.busy) return;
  state.history = [];
  state.context = { document: null, awaiting_document: false };
  messages.replaceChildren();
  welcome.hidden = false;
  prompt.value = "";
  showError("");
  resizeInput();
  prompt.focus();
  window.scrollTo(0, 0);
});

document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => {
    prompt.value = button.dataset.example === "chat" ? "什么是股份回购？请简单解释一下。" :
      "请提取这篇公告的事件字段。\n公告原文：甲公司公告，拟以1亿元收购乙公司100%股权。";
    resizeInput();
    prompt.focus();
  });
});

resizeInput();
refreshHealth();
setInterval(refreshHealth, 30000);
