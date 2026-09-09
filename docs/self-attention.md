# Self-Attention

Self-attention is a mechanism that allows each token in a sequence to **dynamically determine which other tokens are relevant to it and combine information from them**. Each token is transformed into a **Query (Q), Key (K), and Value (V)** representation. Query–Key similarity determines how strongly one token attends to another; these attention weights are then used to compute a weighted combination of the Value vectors. In GPT, the attention is **causal/masked**, meaning a token may attend only to itself and earlier tokens, never future tokens.

---

# Phase 3 — Building Self-Attention from first principles

The most important thing is: **don't start with the famous QKV equation.**

If you start with

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

you can memorize it without understanding it.

Instead, let's **discover why that equation has to exist.**

Think of your current model as having a sequence of token representations:

$$
X =
\begin{bmatrix}
x_1\\
x_2\\
x_3\\
\vdots\\
x_T
\end{bmatrix}
$$

Each \(x_i\) is the embedding/representation of a token.

The problem is:

> **How can \(x_i\) obtain useful information from the other \(x_j\)'s?**

That's the entire reason self-attention exists.

---

## 1. First solve the problem without attention

Suppose our sequence is:

> **The cat sat on the mat**

Imagine we are currently processing **"mat"**.

The representation of `"mat"` initially contains information about `"mat"` itself.

But perhaps we want it to incorporate information from:

> `cat`, `sat`, `on`, `the`

So we need some operation that creates a new representation:

$$
x_{\text{mat}}'
=
f(x_{\text{cat}},x_{\text{sat}},x_{\text{on}},x_{\text{the}},x_{\text{mat}})
$$

In other words:

> **The new representation of a token should depend on the other tokens around it.**

This is called **contextualization**.

An embedding is relatively static:

> `"bank"` has one learned vector.

After self-attention:

> `"bank"` in _river bank_ and `"bank"` in _bank account_ can acquire different contextual representations.

That's a major conceptual transition.

---

# 2. The simplest possible solution: average everything

Before inventing attention, let's use a stupid solution.

For every token, simply average all previous token representations.

For example:

$$
x_4' =
\frac{x_1+x_2+x_3+x_4}{4}
$$

Now token 4 contains information from tokens 1–4.

Great!

But there's a huge problem.

Suppose:

> **The dog chased the cat because it was fast.**

When processing `"it"`, perhaps `"dog"` is highly relevant, while `"the"` is not.

A simple average says:

> "Everybody gets equal importance."

That's obviously inadequate.

So we need:

> **different amounts of information from different tokens.**

And now we're getting close to attention.

---

# 3. Give every token a weight

Instead of:

$$
x_i'=\frac{x_1+x_2+x_3+x_4}{4}
$$

we want:

$$
x_i'
=
w_1x_1+w_2x_2+w_3x_3+w_4x_4
$$

where the weights tell us:

> **How much should token \(i\) care about token \(j\)?**

For example, conceptually:

| Token  | Attention weight |
| ------ | ---------------: |
| The    |             0.05 |
| dog    |             0.55 |
| chased |             0.10 |
| cat    |             0.25 |
| it     |             0.05 |

Now the representation of `"it"` receives much more information from `"dog"` and `"cat"`.

This is the fundamental idea of **attention**.

---

# 4. But where do the weights come from?

This is the crucial question.

We don't want to manually specify:

> dog = 0.55
> cat = 0.25

The neural network must **learn** those relationships.

So we need a mechanism that can ask:

> **"How relevant is token A to token B?"**

The natural mathematical tool is **similarity**.

One simple similarity measure is the dot product:

$$
x_i \cdot x_j
$$

If two vectors point in similar directions, the dot product tends to be larger.

So we could calculate:

$$
x_i\cdot x_1,\quad
x_i\cdot x_2,\quad
x_i\cdot x_3,\ldots
$$

These become relevance scores.

And now we have the beginning of attention.

---

# 5. But raw embeddings aren't enough

Here's the subtle part.

We don't necessarily want:

> "How similar are these two token embeddings?"

We want something more powerful:

> **"Given what I am currently looking for, how relevant is this other token?"**

That leads to the three objects:

### Query

The **Query** represents:

> _What information am I looking for?_

### Key

The **Key** represents:

> _What kind of information do I contain?_

### Value

The **Value** represents:

> _What information should I actually provide if you decide to attend to me?_

This analogy is extremely useful.

Imagine a library.

You walk in looking for:

> **"Books about neural networks."**

Your search request is the **Query**.

Each book has metadata describing what it contains:

> "deep learning, optimization, neural networks"

Those descriptors are the **Keys**.

Once you find relevant books, you actually read their contents.

Those contents are the **Values**.

So:

> **Query finds relevant Keys → Keys determine attention → Values provide information.**

---

# 6. Where do Q, K and V come from?

Here's where the neural network enters.

For each token representation \(x\), we learn three transformations:

$$
q=xW_Q
$$

$$
k=xW_K
$$

$$
v=xW_V
$$

where \(W_Q,W_K,W_V\) are learned weight matrices.

This is extremely important.

**Q, K, and V are not three different kinds of tokens.**

They are **three different learned projections of the same token representation**.

One token therefore becomes:

> token representation → Query
> token representation → Key
> token representation → Value

And because \(W_Q,W_K,W_V\) are learned, the network learns what kinds of relationships are useful.

---

# 7. Now attention emerges naturally

Suppose token \(i\) wants information.

It has a Query:

$$
q_i
$$

Every previous token has a Key:

$$
k_1,k_2,\ldots,k_i
$$

We calculate:

$$
q_i\cdot k_j
$$

for every allowed token \(j\).

This produces relevance scores:

$$
s_{i1},s_{i2},\ldots,s_{ii}
$$

Conceptually:

> **Query asks a question. Keys determine which tokens are relevant.**

Then we turn those scores into normalized weights using softmax.

$$
\alpha_{ij}
=
\operatorname{softmax}(q_i k_j)
$$

Now the weights sum to 1.

Finally, we use those weights to mix the **Values**:

$$
o_i
=
\sum_j \alpha_{ij}v_j
$$

And **that is attention.**

The new representation \(o_i\) is a context-dependent mixture of information from other tokens.

---

# 8. Now understand the famous matrix equation

Once the individual-token mechanism is clear, the famous equation becomes almost obvious:

$$
\boxed{
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
}
$$

Read it **left to right**:

$$
QK^T
$$

Calculate every Query–Key compatibility.

↓

$$
\sqrt{d_k}
$$

Scale the scores so their magnitudes don't become problematic as the dimensionality increases.

↓

### `softmax`

Turn compatibility scores into normalized attention probabilities.

↓

### \(V\)

Use those probabilities to mix the actual information.

So the equation is simply:

> **measure relevance → normalize relevance → retrieve information according to relevance.**

That's all.

---

# 9. Now add the GPT-specific ingredient: causal masking

There's one huge problem.

Suppose you're training:

> `The cat sat on`

to predict:

> `the`

The representation at `"cat"` cannot be allowed to look at `"sat"` or `"on"`.

Otherwise the model cheats.

GPT therefore uses **causal self-attention**.

Token \(t\) can see:

$$
1,2,\ldots,t
$$

but not:

$$
t+1,t+2,\ldots,T
$$

So the attention matrix conceptually looks like:

$$
\begin{bmatrix}
✓&-&-&-&-\\
✓&✓&-&-&-\\
✓&✓&✓&-&-\\
✓&✓&✓&✓&-\\
✓&✓&✓&✓&✓
\end{bmatrix}
$$

This triangular structure is absolutely fundamental to GPT.

**Self-attention + causal masking = autoregressive information flow.**

---

# 10. What you should physically build

For your Pico-GPT, don't think of "self-attention" as one giant object.

Build it mentally as this pipeline:

**Token representations**

↓

**Linear projection into Q, K, V**

↓

**Query–Key dot products**

↓

**Causal mask**

↓

**Scale**

↓

**Softmax**

↓

**Weighted Value aggregation**

↓

**Contextualized token representations**

That is your self-attention mechanism.

Then later:

**multiple attention heads → multi-head attention**

and eventually:

**multi-head attention + FFN + residual connections + LayerNorm → Transformer block.**

---

# The most important experiment for you

Before you build multi-head attention, I want you to understand **single-head causal self-attention with a tiny sequence** completely.

Imagine only four tokens:

> **I love deep learning**

Your task should be to manually trace:

$$
X \rightarrow Q,K,V
$$

then:

$$
QK^T
$$

then apply the causal mask,

then softmax,

then multiply by \(V\),

and finally understand what comes out for each token.

Once you can do that manually, the implementation becomes almost mechanical.

**Don't memorize QKV. Derive it from the problem of information retrieval between tokens.**

That is the Feynman-style connection you want: **we started with "tokens need context," discovered weighted averaging, discovered learned relevance, and Q/K/V emerged as the machinery needed to make that relevance useful.**

For your Pico-GPT, **single-head causal self-attention is the exact next milestone**. After that, we can derive **scaled dot-product attention numerically with a 4-token example**, then turn that into the conceptual architecture you'll implement.

```

```
