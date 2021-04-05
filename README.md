# Prog Integer Parse
(*Programmer's Integer Parsing*)  
This library is meant for when you want to allow the base to be specified be a prefix, while also having a default base if no prefix is specified.  
It does not aim to be super-performant, but it does aim to avoid unneeded heap allocations. Aka no heap allocations because this is just parsing some text into a fixed size value!
  
### Problem:  
`str::parse::<T>() -> Result<T, ParseIntError>` Only supports base-10. Good for normal user-facing applications.  
`T::from_str_radix(text: &str, radix: u32) -> Result<T, ParseIntError>` does not support specifying the base through a prefix.  
This means you have to check the appropriate prefixes yourself. Sure, that isn't too hard. You can literally just do: `let text = &text[2..];` (if they have a prefix).  
The problem you run into is that for *negative* numbers with prefixes you can't just slice. If you have `-0x4a`, then you literally can't slice the prefix out.  
**Options**: Allocate a string; check for sign, strip sign and prefix, and then parse signed values as their unsigned counterpart, then do irritating checking to ensure it is valid.  
  
### Dependencies:
`num-traits`: This is so that we can be generic over integer types. It does not depend on anything else (except at build time, where it depends on `autocfg`).  
Currently the dependency uses some `unsafe`, but at a glance, it is working with floats which this library does not use at all.  
It would be entirely possible, and relatively simple, to recreate definitions of only the functions that are used from `num-traits`.

### Notes
Note: This library only aims to support the standard library integer types. `{u,i}{8,16,32,64,128,size}`. If you want to use it on some more exotic integer type (ex: `u4`), then it *might* work but it may also _not_ work. If you have a problem getting some integer type working, feel free to open an issue, and potential deliberate support could be added.  

### Stability
Currently this library should likely be considered unstable since it was only recently written.  
But, even once the workings become properly stable, it will not move to 1.0 unless `num-traits` moves to `1.0`. Or the parts that are from `num-traits` are rewritten to not depend on it.