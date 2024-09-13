# cwa
Closing Window Attention

### Experimental Results (Sep 12, 2024)

- All shapes of closing windows (differing window sizes per layer) slightly underperform fixed window size (StreamingLLM setting) in PG19 PPL.
  - Linearly increasing
  - Linearly decreasing
  - ㄷ shaped
