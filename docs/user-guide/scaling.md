# Scaling and performance

Cubed is designed to scale to processing large arrays (~TB's) without losing performance.

This page aims to help you understand how Cubed's design scales both in theory and in practice.


## Preface: Types of scaling

- Horizontal versus vertical scaling 
- Weak scaling versus strong scaling
    - Strong scaling is defined as how the solution time varies with the number of processors for a fixed total problem size.
    - Weak scaling is defined as how the solution time varies with the number of processors for a fixed problem size per processor. 

Cubed is designed to scale horizontally across a large number of machines in the cloud with the aim of providing good weak scaling properties when processing even very large array datasets.

## Theoretical vs Practical Scaling of Cubed

### Single-step Calculation

#### **Theoretical Scaling**

- Use rechunk as an example 
- Limited by concurrent writes to Zarr 
- Assuming serverless service provides infinite workers 
- Weak scaling should be totally linear 
- i.e. larger problem completes in same time given larger resources 
- (Compute-bound step would have speedup limited by Ahmdahl's law??)

#### **Practical Performance**
- Actually requires parallelism, so won't happen with single-threaded executor 
- Weak scaling requires more workers than output chunks, might need to set config for big problems 
- Without enough workers strong scaling should be totally linear until more workers than output chunks 
- Stragglers - so turn on backups 
- Failures, once restarted, are basically stragglers 
- Worker start-up Multiple steps 
- Number of steps in plan sets min total execution time 
- So reduce number of steps in plan 
- Reductions can be done in fewer iterative steps if allowed_mem is larger, rule of thumb for this?? 
- Also can fuse steps 
- Happens automatically 
- Can't fuse through rechunking steps without requiring shuffle, which can violate memory constraint in general 
- But multiple blockwise operations can be fused 
- Can multiple rechunk operations be fused? 

### Multi-step Calculation

#### **Theoretical Scaling**

#### **Practical Performance**

### Multi-pipeline Calculation

#### **Theoretical Scaling**
- Two separate arrays you ask to compute simultaneously 
- Just requires enough workers for both 
- Same logic for two arrays which input into single array (or vice versa) 

#### **Practical Performance**
- Currently Cubed will not execute independent pipelines in parallel on all executors 

## Other Performance Considerations

### Different Executors
- Some worker startup time much faster than others 
- Different limits to max workers 
- If you used dask as an executor its scheduler might do unintuitive things - Does Beam have different properties?

### Different Cloud Providers
- GCF worse for stragglers than AWS

## Diagnosing Performance

### Optimized Plan
- The `Plan.visualize()` shows the optimized plan
- Can see how many steps there are

### History Callback
- Can work out how much time was spent in worker startup
- Can also work out how much stragglers affected the overall speed

### Timeline Visualization Callback
- Can visually see the above
- We want vertical lines on this plot

## Tips

- There are very few "magic numbers" in Cubed - means calculations normally take as long as they take, as there are few other parameters to tune
- Use `measure_reserved_mem`
- ~100MB chunks
- ~2GB `allowed_mem` (or larger?)
- Use Cubed only for the part of the calculation you need to
