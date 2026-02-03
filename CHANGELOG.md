# Changelog

<!-- next-header -->

## git

- Now works on stable rust!
  - portable_simd => wide
  - refactors/simplifications to drop `const_generics` requirement
- API cleanup (make some stuff private; expose some type aliases)
- Make `RankerT::rank` method unsafe and rename to `rank_unchecked`.

## 0.1.0

- Initial release.
- Not yet ready for general usage because of nightly features, but claiming the name.
