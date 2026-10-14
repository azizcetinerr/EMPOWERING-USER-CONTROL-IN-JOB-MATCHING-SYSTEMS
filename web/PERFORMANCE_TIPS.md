# Performance Tips

## Problem: Too Slow or Too Fast (429 Error)

### Solution: Adaptive Rate Limiting ✅

The system now adjusts its speed automatically:
- **On a 429 error**: slows down automatically
- **Successful requests**: speeds up automatically
- **Optimal speed**: runs at the maximum speed the server allows

## Expected Durations

### New projects (first time)
- **100 projects**: ~15-20 minutes (with adaptive rate limiting)
- **50 projects**: ~8-10 minutes
- **10 projects**: ~2-3 minutes

### Existing projects (from the cache)
- **CV analysis only**: ~2-3 minutes
- **If there are no new projects**: ~1-2 minutes

## Speed-Up Tips

### 1. Use the existing projects
If project data was collected before:
- Only new projects are scraped
- It runs much faster

### 2. Adjust the number of threads
In `config/config.yaml`:
```yaml
performance:
  num_workers: 4  # Can be increased: 5-6 (but the 429 risk increases)
```

**Recommendations**:
- Good internet: 5-6 threads
- Average internet: 4 threads (default)
- Slow internet: 3 threads

### 3. Use the first N projects
To use only the first N projects instead of all projects:
- Limit the number of projects in the pipeline
- Or add a "top 50 projects only" option to the web interface

## Troubleshooting

### Too many 429 errors
**Symptoms**: constant "429" errors in the logs
**Solution**:
- Reduce the number of threads: `num_workers: 3`
- Increase the request delay: `request_delay: 2`

### Very slow progress
**Symptoms**: waiting for 1 hour+
**Solution**:
- Increase the number of threads: `num_workers: 5-6`
- Decrease the request delay: `request_delay: 0.5`
- Use the existing projects (from the cache)

### I do not see any progress
**Symptoms**: nothing in the log
**Solution**:
- Increase the log level: `logging.level: DEBUG`
- Check the terminal output

## Optimal Settings

### Fast internet + powerful server
```yaml
performance:
  num_workers: 6

scraping:
  requests:
    request_delay: 0.5
```

### Normal internet
```yaml
performance:
  num_workers: 4  # Default

scraping:
  requests:
    request_delay: 1  # Default
```

### Slow internet or rate limiting problems
```yaml
performance:
  num_workers: 3

scraping:
  requests:
    request_delay: 2
    retry_delay: 15
```

## How Does Adaptive Rate Limiting Work?

1. **Start**: starts with a 1-second delay
2. **429 error**: increases the delay 1.5x (max 5 seconds)
3. **10 successful requests**: decreases the delay 0.9x (min 1 second)
4. **Optimal speed**: runs at the maximum speed the server allows

This way:
- ✅ Rate limiting is avoided
- ✅ It runs as fast as possible
- ✅ It adjusts automatically
