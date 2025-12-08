import matplotlib.pyplot as plt

# Market + strategy prices
mid = 100.025
bid = 100.00
ask = 100.05

baseline_bid = 100.01
baseline_ask = 100.04

ofi_bid = 100.02
ofi_ask = 100.06

future_price = 100.08

fig, ax = plt.subplots(figsize=(6, 6))

# --- Market bid/ask + mid ---
ax.hlines(bid, 0.5, 1.5, linestyles="--")
ax.text(0.48, bid, "Market Bid 100.00 @ 500", va="center", ha="right")

ax.hlines(ask, 0.5, 1.5, linestyles="--")
ax.text(0.48, ask, "Market Ask 100.05 @ 300", va="center", ha="right")

ax.hlines(mid, 0.3, 1.7, linestyles=":")
ax.text(0.48, mid, "Mid 100.025", va="center", ha="right")

# --- Baseline quotes (x = 1.0) ---
ax.scatter([1, 1], [baseline_bid, baseline_ask], s=60)
ax.text(1.02, baseline_bid, "Baseline Bid 100.01", va="center", ha="left")
ax.text(1.02, baseline_ask, "Baseline Ask 100.04", va="center", ha="left")

# --- OFI quotes (x = 1.4) ---
ax.scatter([1.4, 1.4], [ofi_bid, ofi_ask], s=60, marker="s")
ax.text(1.42, ofi_bid, "OFI Bid 100.02", va="center", ha="left")
ax.text(1.42, ofi_ask, "OFI Ask 100.06", va="center", ha="left")

# --- Future price (x = 1.7) ---
ax.scatter([1.7], [future_price], s=70, marker="^")
ax.text(1.72, future_price, "Future Price 100.08", va="center", ha="left")

# --- Annotations ---
ax.annotate(
    "OFI = +0.5\nBuying pressure → shift quotes up",
    xy=(1.2, 100.045),
    xytext=(0.6, 100.09),
    arrowprops=dict(arrowstyle="->"),
    ha="left",
)

ax.annotate(
    "Baseline sells too low\n(adverse selection)",
    xy=(1.0, baseline_ask),
    xytext=(0.4, 99.99),
    arrowprops=dict(arrowstyle="->"),
    ha="left",
)

ax.annotate(
    "OFI strategy avoids fill\nand waits for better price",
    xy=(1.4, ofi_ask),
    xytext=(1.1, 100.10),
    arrowprops=dict(arrowstyle="->"),
    ha="left",
)

ax.set_xlim(0.2, 1.9)
ax.set_ylim(99.97, 100.11)
ax.set_xticks([])
ax.set_ylabel("Price")
ax.set_title("Using OFI to Shift Quotes and Avoid Selling Too Cheap")

plt.tight_layout()
plt.show()