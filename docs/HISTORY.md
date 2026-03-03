# Metricate

## Why

It's not always trivial to say whether one clustering is better than another.

Sometimes it is. You look at two plots and one is clearly better:

![Good clustering](assets/clustering_good.png)
*Clean separation, each color stays together*

![Bad clustering](assets/clustering_bad.png)
*Same data, worse clustering - colors mixed everywhere*

Sometimes the difference is just obvious.

[INSERT EXAMOPLE HERE]

But usually it's not that simple. You have two clusterings, they both look... fine? Different, but reasonable. Which one do you pick?

That's why metrics exist. Silhouette, Davies-Bouldin, Calinski-Harabasz, etc. They try to quantify "good".

Problem is there are like 40+ of them. And some are [mathematically equivalent](https://www.cell.com/action/showFullTableHTML?isHtml=true&tableId=tbl3&pii=S2405-8440%2825%2900333-0) to each other. You run all of them and get 40 different opinions.

## Step 1: Naive Winner

Simplest thing you can do: assume all metrics are equally informative and important.

Run every metric. Count wins. Clustering A wins 25 metrics, clustering B wins 18? A is better.

This is what Metricate does right now. This is what the web UI shows you. It's transparent and it works surprisingly well.

But what if A wins 16 and B wins 17?

Is that real? Or noise?

The equal-weights assumption breaks down fast.

## Step 2: Learn the Weights

Some metrics are more useful than others. Some are redundant. Some might even be misleading.

So learn which ones matter.

How:
1. **Positive set** - start with known good clusterings
2. **Negative set** - degrade those clusterings in lots of ways (swap labels randomly, merge clusters, split clusters, remove points, add noise, etc)
3. **Train** - model learns which metrics actually distinguish good from bad

The degradation framework is done. We can generate negative examples across a bunch of degradation types:
- Label swaps (random, neighboring clusters, distant clusters)  
- Merges and splits
- Point removal (random, core points, boundary points)
- Embedding perturbations

Each one makes the clustering measurably worse. Perfect training signal.

## What's Next

Learned weights. A single score that tells you which clustering wins, and how confident we are.

Not 40 numbers. Not "it depends". Just: this one.
