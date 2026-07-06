export interface ConfigView {
  board_size: number;
  max_walls: number;
  max_steps: number;
  defaults: {
    mcts_n: number;
    mcts_c_puct: number;
    temperature: number | null;
    mcts_noise_epsilon: number;
    mcts_noise_alpha: number | null;
    leaf_parallelism: number;
    virtual_loss: number;
    mcts_worker_threads: number | null;
  };
}
export interface ModelsView { models: string[]; default: string | null }

async function getJson<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return (await r.json()) as T;
}

export const fetchConfig = () => getJson<ConfigView>("/api/config");
export const fetchModels = () => getJson<ModelsView>("/api/models");
