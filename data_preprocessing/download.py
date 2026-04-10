from datasets import load_dataset
ds1 = load_dataset("yixuantt/FinEntity")
ds1.save_to_disk("FinEntity")
from datasets import load_dataset
ds2 = load_dataset("baptle/financial_headlines_market_based")
ds2.save_to_disk("FinMarBa")