const SAMPLES = [
  {
    keys: ["年假", "休假", "请假"],
    question: "年假如何申请？",
    answer:
      "入职满 1 年可享 5 天年假，满 3 年增至 10 天。请在 OA「人事 - 休假申请」中提交，至少提前 3 个工作日，并由直属主管审批。",
    citations: [
      ["员工手册", "4.2 年假与调休"],
      ["OA 操作指南", "休假申请流程"],
    ],
  },
  {
    keys: ["报销", "发票", "差旅", "费用"],
    question: "报销流程是什么？",
    answer:
      "在 OA「财务 - 费用报销」提交发票与行程单，经部门负责人审批后由财务复核。建议在费用发生后 30 天内提交，每月 25 日前的单据计入当月批次。",
    citations: [
      ["财务报销制度", "3.1 报销时限"],
      ["费用管理规范", "附录 A 报销材料"],
    ],
  },
  {
    keys: ["试用期", "转正", "入职"],
    question: "试用期多久？",
    answer:
      "标准试用期为 3 个月，试用期薪资为转正薪资的 80%。转正前 2 周需在 OA 提交转正总结，由直属主管与 HR 共同评估。",
    citations: [
      ["员工手册", "2.3 试用期管理"],
      ["转正操作指引", "评估与审批"],
    ],
  },
  {
    keys: ["远程", "居家", "办公"],
    question: "可以远程办公吗？",
    answer:
      "支持每周最多 2 天远程办公，需提前 1 个工作日在 OA 提交申请，并确保当日核心会议可线上参加。涉及保密数据的岗位需额外审批。",
    citations: [
      ["远程办公管理办法", "2.1 适用范围"],
      ["信息安全规范", "5.4 远程访问"],
    ],
  },
  {
    keys: ["采购", "买", "预算"],
    question: "采购需要走什么流程？",
    answer:
      "单笔 5,000 元以下由部门负责人审批；5,000 元以上需提交采购申请并附至少 2 家比价。IT 类采购还需信息部会签，完成后由采购统一下单。",
    citations: [
      ["采购管理制度", "3.2 审批权限"],
      ["供应商管理办法", "4.1 比价要求"],
    ],
  },
];

const chatBody = document.querySelector("#chat-body");
const form = document.querySelector("#chat-form");
const input = document.querySelector("#question");
const submitButton = form.querySelector("button[type='submit']");
const suggestions = document.querySelectorAll("[data-question]");

let busy = false;

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (char) => {
    const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
    return map[char];
  });
}

function scrollToBottom() {
  chatBody.scrollTo({ top: chatBody.scrollHeight, behavior: "smooth" });
}

function addMessage(role, html, meta = "") {
  const message = document.createElement("div");
  message.className = `message ${role}`;
  message.innerHTML = `<div class="bubble">${html}</div>${meta ? `<p class="message-meta">${meta}</p>` : ""}`;
  chatBody.appendChild(message);
  scrollToBottom();
  return message;
}

function citationMarkup(citations) {
  return `<div class="citations">${citations
    .map(
      ([title, section], index) =>
        `<div class="citation"><b>${index + 1}</b><span>${title} · ${section}</span></div>`
    )
    .join("")}</div>`;
}

function findSample(question) {
  const normalized = question.replace(/\s/g, "").toLowerCase();
  return SAMPLES.find((sample) => sample.keys.some((key) => normalized.includes(key.toLowerCase())));
}

function answerFor(question) {
  const sample = findSample(question);
  if (sample) {
    return {
      html: `<p>${sample.answer}</p>${citationMarkup(sample.citations)}`,
      meta: "Router → Retrieve → Generate → Reflect · 4 个 Agent 已完成",
    };
  }
  return {
    html:
      "<p>这个静态演示只覆盖了年假、报销、试用期、远程办公和采购几个示例问题。完整系统会先从企业知识库检索相关片段，再基于证据生成答案。</p>",
    meta: "Demo 知识库未命中 · 未调用真实模型",
  };
}

function submitQuestion(question) {
  const trimmed = question.trim();
  if (!trimmed || busy) return;

  busy = true;
  submitButton.disabled = true;
  addMessage("user", `<p>${escapeHtml(trimmed)}</p>`);

  const typing = document.createElement("div");
  typing.className = "message assistant";
  typing.innerHTML = `<div class="bubble"><span class="typing" aria-label="正在检索"><i></i><i></i><i></i></span></div>`;
  chatBody.appendChild(typing);
  scrollToBottom();

  window.setTimeout(() => {
    typing.remove();
    const answer = answerFor(trimmed);
    addMessage("assistant", answer.html, answer.meta);
    busy = false;
    submitButton.disabled = false;
    input.focus();
  }, 720);
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value;
  input.value = "";
  submitQuestion(question);
});

suggestions.forEach((button) => {
  button.addEventListener("click", () => {
    if (busy) return;
    submitQuestion(button.dataset.question || button.textContent || "");
  });
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});
