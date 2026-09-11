# Self-Attention input and whole process

You now have the correct starting point for self-attention. Your embedded input

$$
Xte : Embedding\ tensor\ of\ X
$$

has shape \([W,T,d]=[85,8,8]\): 85 context windows, each containing 8 token positions, with each token represented by an 8-dimensional vector. Self-attention now operates **independently across the 8 positions within each window**, allowing every token to determine which earlier/current tokens are relevant and aggregate their information. The target tensor \(Y_t\) remains token IDs and is **not embedded** because it represents the desired next-token classes.

# Step 4 — From Xte to Self-Attention

You have:

$$
\boxed{X_{te}\in\mathbb{R}^{W\times T\times d}}
$$

with:

$$
W=85,\qquad T=8,\qquad d=8
$$

so:

$$
X_{te}\in\mathbb{R}^{85\times8\times8}
$$

This is excellent. **Now we should forget about the 85 windows temporarily.**

Take just **one window**.

Then:

$$
X\in\mathbb{R}^{8\times8}
$$

Think of it as:

```text
          embedding dimensions
       ←────────────────────────→
token 1   [ .  .  .  .  .  .  .  . ]
token 2   [ .  .  .  .  .  .  .  . ]
token 3   [ .  .  .  .  .  .  .  . ]
token 4   [ .  .  .  .  .  .  .  . ]
token 5   [ .  .  .  .  .  .  .  . ]
token 6   [ .  .  .  .  .  .  .  . ]
token 7   [ .  .  .  .  .  .  .  . ]
token 8   [ .  .  .  .  .  .  .  . ]
```

The **rows are tokens/positions**.
The **columns are features of their representations**.

Now we need to make these token representations communicate.

---

# 4.1 The first question: how does one token communicate with another?

Suppose we're processing token 8.

It currently has its own representation:

$$
x_8\in\mathbb{R}^{8}
$$

But perhaps useful information exists in:

$$
x_1,x_2,\ldots,x_7
$$

We want token 8 to determine:

> **Which of these previous tokens are relevant to me?**

This is the fundamental problem self-attention solves.

So we need a way for token 8 to **ask a question** and for every other token to indicate whether it contains relevant information.

This is where **Query, Key, Value** emerge.

---

# 4.2 Create Query, Key and Value

We take your \(X\) and create three different learned projections:

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

Here:

$$
W_Q,\ W_K,\ W_V
$$

are **learnable matrices**.

Because your current embedding dimension is:

$$
d=8
$$

we could initially make:

$$
W_Q,W_K,W_V\in\mathbb{R}^{8\times8}
$$

Therefore:

$$
X:[8,8]
$$

becomes:

$$
Q:[8,8]
$$

$$
K:[8,8]
$$

$$
V:[8,8]
$$

And with all 85 windows:

$$
X_{te}:[85,8,8]
$$

becomes:

$$
Q,K,V:[85,8,8]
$$

---

# 4.3 But what do Q, K and V actually mean?

Don't think of them as mysterious Transformer terminology.

For each token:

**Query:** "What information am I looking for?"

**Key:** "What kind of information do I contain?"

**Value:** "What information do I provide if you find me relevant?"

For example, imagine token 8 is asking:

> "Who or what earlier token is relevant to my current prediction?"

Its Query gets compared against the Keys of the previous tokens.

Conceptually:

$$
q_8\cdot k_1
$$

$$
q_8\cdot k_2
$$

$$
q_8\cdot k_3
$$

...

$$
q_8\cdot k_8
$$

These numbers measure **compatibility/relevance**.

---

# 4.4 This creates the attention-score matrix

Here's the first important matrix operation:

$$
\boxed{S=QK^T}
$$

For one context window:

$$
Q:[8,8]
$$

and:

$$
K^T:[8,8]
$$

therefore:

$$
S:[8,8]
$$

So you get:

```text
              Keys
          1   2   3   4   5   6   7   8
       ┌──────────────────────────────────
Query 1│
Query 2│
Query 3│
Query 4│
Query 5│
Query 6│
Query 7│
Query 8│
```

Each cell \(S\_{ij}\) means:

> **How strongly should token \(i\)'s Query relate to token \(j\)'s Key?**

For example:

$$
S_{8,3}=q_8\cdot k_3
$$

means:

> How relevant is token 3 to token 8?

This is the heart of self-attention.

---

# 4.5 But GPT has one critical restriction

Your model is autoregressive.

Token 8 can see:

$$
1,2,3,4,5,6,7,8
$$

Token 5 can see:

$$
1,2,3,4,5
$$

but **cannot see 6, 7, 8**.

Therefore your score matrix must become:

$$
\begin{bmatrix}
s_{11}&-\infty&-\infty&-\infty&-\infty&-\infty&-\infty&-\infty\\
s_{21}&s_{22}&-\infty&-\infty&-\infty&-\infty&-\infty&-\infty\\
s_{31}&s_{32}&s_{33}&-\infty&-\infty&-\infty&-\infty&-\infty\\
\vdots&\vdots&\vdots&\ddots& & & &\\
s_{81}&s_{82}&s_{83}&s_{84}&s_{85}&s_{86}&s_{87}&s_{88}
\end{bmatrix}
$$

The `-\infty` isn't arbitrary.

Later, when softmax is applied:

$$
e^{-\infty}=0
$$

so those forbidden future positions receive **zero attention probability**.

This is your **causal mask**.

---

# 4.6 Then turn scores into probabilities

Raw scores aren't probabilities.

We therefore apply scaling and softmax:

$$
A=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)
$$

For your first experiment, if:

$$
d_k=8
$$

then:

$$
\sqrt{d_k}=\sqrt8
$$

The resulting:

$$
A:[8,8]
$$

contains attention weights.

For example, one row might conceptually become:

$$
[0.05,\ 0.10,\ 0.20,\ 0.15,\ 0.50,\ 0,\ 0,\ 0]
$$

Meaning:

> this token receives 5% of its information from position 1, 10% from position 2, 20% from position 3, etc.

The future positions are zero because of causal masking.

---

# 4.7 Finally retrieve information from V

Now comes the final operation:

$$
\boxed{O=AV}
$$

You have:

$$
A:[8,8]
$$

and:

$$
V:[8,8]
$$

so:

$$
O:[8,8]
$$

Every output token is now a **weighted mixture of information from the tokens it was allowed to see**.

This is why I want you to think of self-attention as:

$$
\boxed{
\text{decide relevance}
\rightarrow
\text{normalize relevance}
\rightarrow
\text{mix information}
}
$$

---

# The complete mechanism

For **one** of your 85 windows:

$$
X:[8,8]
$$

↓

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

↓

$$
Q,K,V:[8,8]
$$

↓

$$
QK^T
$$

↓

$$
[8,8]\text{ attention scores}
$$

↓

**causal mask**

↓

**scale**

↓

**softmax**

↓

$$
A:[8,8]
$$

↓

$$
AV
$$

↓

$$
\boxed{O:[8,8]}
$$

And for your complete dataset:

$$
\boxed{
[85,8,8]
\rightarrow
[85,8,8]
}
$$

The shape stays the same, but the **content changes**: each token representation becomes contextualized by the other permitted tokens.

---

# One thing I want you to do before coding

Don't implement the whole thing yet.

Take **one** window:

$$
X_{te}[0]
$$

which is:

$$
[8,8]
$$

and understand only this next transformation:

$$
\boxed{X\rightarrow Q,K,V}
$$

Specifically, understand why:

$$
[8,8]\times[8,8]=[8,8]
$$

and what **each row** and **each column** means.

Once that is solid, the next step is extremely concrete: we'll take your actual \(8\times8\) embedded matrix and manually walk through **\(QK^T\)**, including exactly why its result is an \(8\times8\) matrix and what every cell represents.
