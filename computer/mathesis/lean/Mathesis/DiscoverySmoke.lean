import Mathlib

namespace Mathesis

example (x y : ℝ) :
    (x + y)^3 = x^3 + 3*x^2*y + 3*x*y^2 + y^3 := by
  ring

example (x : ℝ) : x^2 ≥ 0 := by
  exact sq_nonneg x

example : Nat.gcd 84 30 = 6 := by
  decide

example : Nat.choose 8 3 = 56 := by
  decide

example (a b : ℚ) (h : b ≠ 0) : (a / b) * b = a := by
  field_simp

end Mathesis
