//! Program cache for compiled kernels
//!
//! Uses DSL decompile output as cache key for deterministic graph identification.

use std::collections::HashMap;

/// Cache statistics
#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: usize,
    /// Number of cache misses
    pub misses: usize,
}

impl CacheStats {
    /// Calculate hit rate (0.0 to 1.0)
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Program cache using decompile output as key
pub struct ProgramCache<P> {
    cache: HashMap<String, P>,
    stats: CacheStats,
    enabled: bool,
}

impl<P> Default for ProgramCache<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P> ProgramCache<P> {
    /// Create a new empty cache
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            stats: CacheStats::default(),
            enabled: true,
        }
    }

    /// Enable or disable caching
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if caching is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get a cached program by key
    pub fn get(&mut self, key: &str) -> Option<&P> {
        if !self.enabled {
            return None;
        }

        if let Some(program) = self.cache.get(key) {
            self.stats.hits += 1;
            Some(program)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Insert a program into the cache
    pub fn insert(&mut self, key: String, program: P) {
        if !self.enabled {
            return;
        }

        self.cache.insert(key, program);
    }

    /// Clear all cached programs
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = CacheStats::default();
    }

    /// Get number of cached entries
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_basic() {
        let mut cache: ProgramCache<String> = ProgramCache::new();

        // Miss
        assert!(cache.get("key1").is_none());
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 0);

        // Insert
        cache.insert("key1".to_string(), "program1".to_string());
        assert_eq!(cache.len(), 1);

        // Hit
        assert_eq!(cache.get("key1"), Some(&"program1".to_string()));
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn test_cache_disabled() {
        let mut cache: ProgramCache<String> = ProgramCache::new();
        cache.set_enabled(false);

        cache.insert("key1".to_string(), "program1".to_string());
        assert!(cache.is_empty());

        assert!(cache.get("key1").is_none());
        // Stats should not change when disabled
        assert_eq!(cache.stats().misses, 0);
    }

    #[test]
    fn test_cache_clear() {
        let mut cache: ProgramCache<String> = ProgramCache::new();
        cache.insert("key1".to_string(), "program1".to_string());
        cache.insert("key2".to_string(), "program2".to_string());
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_hit_rate() {
        let mut cache: ProgramCache<String> = ProgramCache::new();
        cache.insert("key1".to_string(), "program1".to_string());

        // 1 hit, 1 miss
        let _ = cache.get("key1"); // hit
        let _ = cache.get("key2"); // miss

        assert!((cache.stats().hit_rate() - 0.5).abs() < 0.001);
    }
}
