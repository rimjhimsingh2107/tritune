import type { ReactNode } from "react";

const KW = new Set(("def return for in if else elif while import from as and or not None True False " +
  "with assert lambda pass break continue class").split(" "));
const TOKEN = /(#.*$)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|\b(\d+\.?\d*(?:e-?\d+)?)\b|\b(tl\.\w+|triton\.\w+)\b|\b([A-Za-z_]\w*)\b/g;

function highlight(line: string): ReactNode[] {
  const out: ReactNode[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  TOKEN.lastIndex = 0;
  while ((m = TOKEN.exec(line))) {
    if (m.index > last) out.push(line.slice(last, m.index));
    const [tok, com, str, n, tl, word] = m;
    const cls = com ? "tk-com" : str ? "tk-str" : n ? "tk-num" : tl ? "tk-tl" : word && KW.has(word) ? "tk-kw" : null;
    out.push(cls ? <span key={m.index} className={cls}>{tok}</span> : tok);
    last = m.index + tok.length;
  }
  if (last < line.length) out.push(line.slice(last));
  return out;
}

export function CodeView({ source }: { source: string }) {
  return (
    <pre className="code">
      {source.split("\n").map((l, i) => (
        <span className="line" key={i}><span className="ln">{i + 1}</span>{highlight(l)}</span>
      ))}
    </pre>
  );
}

type Op = { kind: "same" | "add" | "del"; text: string };

// plain LCS line diff; candidate modules are ~100 lines so O(n*m) is fine
export function lineDiff(a: string[], b: string[]): Op[] {
  const n = a.length, m = b.length;
  const dp = Array.from({ length: n + 1 }, () => new Array<number>(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--)
    for (let j = m - 1; j >= 0; j--)
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const ops: Op[] = [];
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) { ops.push({ kind: "same", text: a[i] }); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) ops.push({ kind: "del", text: a[i++] });
    else ops.push({ kind: "add", text: b[j++] });
  }
  while (i < n) ops.push({ kind: "del", text: a[i++] });
  while (j < m) ops.push({ kind: "add", text: b[j++] });
  return ops;
}

export function DiffView({ before, after }: { before: string; after: string }) {
  const ops = lineDiff(before.split("\n"), after.split("\n"));
  const changed = ops.some((o) => o.kind !== "same");
  if (!changed) return <div className="sub">identical source</div>;
  // show changed lines with 2 lines of context
  const keep = ops.map((_, idx) => ops.slice(Math.max(0, idx - 2), idx + 3).some((x) => x.kind !== "same"));
  return (
    <pre className="code">
      {ops.map((o, idx) =>
        keep[idx] ? (
          <span key={idx} className={o.kind === "same" ? "line" : o.kind}>
            <span className="ln">{o.kind === "add" ? "+" : o.kind === "del" ? "−" : " "}</span>{highlight(o.text)}
          </span>
        ) : keep[idx - 1] ? <span key={idx} className="line"><span className="ln">⋯</span></span> : null,
      )}
    </pre>
  );
}
