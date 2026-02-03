import matplotlib.pyplot as plt

from collections import Counter
from datetime import datetime

# Each attendance record is one row: "<m/d/YYYY> <HH:MM:SS>"
# We count number of rows per date (assumes at least 1 row per date).
counts = Counter()
with open("att.csv", "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        date_str = line.split()[0]
        d = datetime.strptime(date_str, "%m/%d/%Y").date()
        counts[d] += 1

dates = sorted(counts.keys())
att = [counts[d] for d in dates]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(dates, att, marker="o")
ax.set_xlabel("Date")
ax.set_ylabel("Attendance")
ax.set_title("Attendance by Date")

# Label each date on the x axis
ax.set_xticks(dates)
ax.set_xticklabels([d.strftime("%m/%d/%Y").lstrip("0").replace("/0", "/") for d in dates], rotation=45, ha="right")

fig.tight_layout()
fig.savefig("att.png", dpi=600)