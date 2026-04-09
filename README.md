# IM-ILP: Optimization-Based Process Discovery

This repository implements the **IM-ILP framework**, a scalable approach to process discovery that reformulates the search for process tree cuts as a mathematical optimization problem. By leveraging integer linear programming, the tool identifies optimal process structures while bypassing the exponential complexity of traditional exhaustive search methods.

---

### 🧬 Core Logic: Recursive Binary Partitioning
The framework adopts the divide-and-conquer architecture of the [**Inductive Miner** family](https://doi.org/10.1007/978-3-319-06257-0_6) but is strictly optimized for **binary cuts**.

* **Recursive Decomposition**: The set of activities from an event log is recursively partitioned into two disjoint subsets.
* **Tree Formation**: This process continues until the algorithm reaches base cases (e.g., single activities or empty traces), resulting in a sound, block-structured **Process Tree**.
* **Structural Soundness**: Every discovered model is mathematically guaranteed to be free from deadlocks and livelocks.

---

### ⚙️ The Mathematical Engine: ILP & Graph Partitioning
Instead of checking every possible combination of activities, IM-ILP utilizes **Integer Linear Programming** with graph partitioning and multi-commodity flow constraints to find the mathematically optimal cut.

The framework evaluates four distinct cut types to identify the best behavioral relationship between activity subsets:

* **Sequence ($\rightarrow$)**: Models a strict chronological and causal order between two activity sets.
* **Exclusive Choice ($\times$)**: Models a strict either-or relationship where activities in one subset never occur with the other in the same trace.
* **Parallel ($\wedge$)**: Represents concurrent behavior, allowing activities from both subsets to execute in any interleaved order.
* **Loop ($\circlearrowleft$)**: Models iterative behavior consisting of a mandatory loop body and a redo part that enables repeated execution.

#### **Customizable Cost Functions**
Current implementation uses cost functions from the **IMbi** framework as objective functions. However, because the per-edge behavioral penalties (deviating and missing costs) are calculated outside the ILPs, the "cost" is fully customizable for the sequence, exclusive choice, and parallel operators.

---

### 🚀 Usage

```python
import pm4py
# Add your discovery logic here
```

---

*For detailed information regarding mathematical formulations, structural reachability constraints, and experimental results, please refer to the full Master Thesis: **"Integrating Graph Partitioning Techniques to Reduce the Computational Cost of Cut Search Algorithms"**.*
