// Минимальный React без сборки: грузим ESM из CDN
import React from "https://esm.sh/react@18.3.1";
import ReactDOM from "https://esm.sh/react-dom@18.3.1/client";

const { useState, useMemo } = React;

function Badge({ label, tone = "neutral" }) {
  const cls = ["badge", `badge-${tone}`].join(" ");
  return React.createElement("span", { className: cls }, label);
}

function HistoryList({ items }) {
  if (!items.length) {
    return React.createElement("p", { className: "muted" }, "История запросов появится здесь");
  }
  return React.createElement(
    "ul",
    { className: "history" },
    items.map((item, idx) =>
      React.createElement(
        "li",
        { key: idx },
        React.createElement("div", { className: "history-text" }, item.text),
        React.createElement(
          "div",
          { className: "history-meta" },
          React.createElement(Badge, {
            label: item.label === "spam" ? "spam" : "ham",
            tone: item.label === "spam" ? "danger" : "success",
          }),
          React.createElement(
            "span",
            { className: "muted" },
            `P(spam) = ${(item.proba * 100).toFixed(1)}%`
          )
        )
      )
    )
  );
}

function App() {
  const [text, setText] = useState("Win a free iPhone if you reply NOW!");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const maxHistory = 8;
  const historySubtitle = history.length
    ? `последние ${Math.min(history.length, maxHistory)}`
    : "нет запросов";

  const sentimentTone = useMemo(() => {
    if (!result) return "neutral";
    return result.label === "spam" ? "danger" : "success";
  }, [result]);

  async function handleSubmit(evt) {
    evt.preventDefault();
    setError("");
    setLoading(true);
    try {
      const resp = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!resp.ok) {
        throw new Error(`Ошибка API: ${resp.status}`);
      }
      const data = await resp.json();
      setResult(data);
      setHistory((prev) =>
        [{ text, label: data.label, proba: data.proba_spam }, ...prev].slice(0, maxHistory)
      );
    } catch (err) {
      setError(err.message || "Неизвестная ошибка");
    } finally {
      setLoading(false);
    }
  }

  function useSample(msg) {
    setText(msg);
  }

  return React.createElement(
    "main",
    { className: "page" },
    React.createElement("header", { className: "hero" }, [
      React.createElement("div", { className: "hero-text", key: "text" }, [
        React.createElement("p", { className: "eyebrow", key: "eyebrow" }, "Demo • SMS Spam Detector"),
        React.createElement("h1", { key: "title" }, "Быстрая проверка SMS на спам"),
        React.createElement(
          "p",
          { className: "lede", key: "lede" },
          "Модель RandomForest на ручных признаках. Введите текст и получите вероятность спама."
        ),
        React.createElement(
          "div",
          { className: "hero-actions", key: "actions" },
          React.createElement("a", { href: "/metrics", target: "_blank", rel: "noreferrer", className: "link" }, "Метрики Prometheus"),
          React.createElement("a", { href: "https://github.com/", target: "_blank", rel: "noreferrer", className: "link" }, "Репозиторий")
        ),
      ]),
      React.createElement(
        "div",
        { className: "card panel", key: "result" },
        React.createElement("p", { className: "muted" }, "Результат"),
        result
          ? React.createElement(
              "div",
              { className: "result" },
              React.createElement(Badge, { label: result.label, tone: sentimentTone }),
              React.createElement(
                "div",
                { className: "proba" },
                React.createElement("span", null, "P(spam)"),
                React.createElement("strong", null, `${(result.proba_spam * 100).toFixed(1)}%`)
              ),
              result.model_path
                ? React.createElement("p", { className: "muted tiny" }, `Модель: ${result.model_path}`)
                : null
            )
          : React.createElement("p", { className: "muted" }, "Отправьте текст, чтобы увидеть результат")
      ),
    ]),
    React.createElement(
      "section",
      { className: "grid" },
      React.createElement(
        "form",
        { className: "panel", onSubmit: handleSubmit },
        React.createElement("label", { className: "label", htmlFor: "text" }, "Введите текст сообщения"),
        React.createElement("textarea", {
          id: "text",
          value: text,
          onChange: (e) => setText(e.target.value),
          rows: 6,
          placeholder: "Пример: Win a free iPhone if you reply NOW!",
        }),
        React.createElement(
          "div",
          { className: "actions" },
          React.createElement(
            "button",
            { type: "button", className: "ghost", onClick: () => useSample("Hey, are we still on for dinner tonight?") },
            "Ham пример"
          ),
          React.createElement(
            "button",
            { type: "button", className: "ghost", onClick: () => useSample("URGENT! You have won $1000. Click here to claim.") },
            "Spam пример"
          ),
          React.createElement(
            "button",
            { type: "submit", disabled: loading || !text.trim() },
            loading ? "Отправляем..." : "Проверить"
          )
        ),
        error && React.createElement("p", { className: "error" }, error)
      ),
      React.createElement(
        "div",
        { className: "panel history-panel" },
        React.createElement("div", { className: "label-row" }, [
          React.createElement("span", { className: "label" }, "История запросов"),
          React.createElement("span", { className: "muted tiny" }, historySubtitle),
        ]),
        React.createElement(HistoryList, { items: history })
      )
    )
  );
}

const rootEl = document.getElementById("root");
const root = ReactDOM.createRoot(rootEl);
root.render(React.createElement(App));
