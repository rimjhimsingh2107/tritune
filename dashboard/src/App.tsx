import { api } from "./api";
import { ComparePage } from "./components/ComparePage";
import { ExperimentList } from "./components/ExperimentList";
import { ExperimentView } from "./components/ExperimentView";
import { useAsync } from "./hooks";
import { useRoute } from "./router";

function Logo() {
  return (
    <svg viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="7" fill="var(--panel-2)" />
      <path d="M6 22 L12 14 L17 18 L26 8" fill="none" stroke="var(--amber)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="26" cy="8" r="3" fill="var(--teal)" />
    </svg>
  );
}

export default function App() {
  const route = useRoute();
  const env = useAsync(() => api.env(), []);
  const [page, id] = route.path;
  return (
    <>
      <header className="top">
        <a className="brand" href="#/"><Logo />TriTune</a>
        <nav>
          <a href="#/" className={!page || page === "exp" ? "on" : ""}>experiments</a>
          <a href="#/compare" className={page === "compare" ? "on" : ""}>compare</a>
        </nav>
        <div className="spacer" />
        {env.data && (env.data.simulated
          ? <span className="badge sim" title="no CUDA device: simulated backend">SIMULATED BACKEND</span>
          : <span className="badge real">{env.data.environment.gpu_name}</span>)}
        {env.data && <span className="badge">planner: {env.data.llm_provider === "none" ? "heuristic" : env.data.llm_provider}</span>}
      </header>
      <main>
        {page === "exp" && id ? <ExperimentView id={id} />
          : page === "compare" ? <ComparePage a={route.query.get("a")} b={route.query.get("b")} />
          : <ExperimentList />}
      </main>
    </>
  );
}
