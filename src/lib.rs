#![forbid(unsafe_code)]
#![warn(clippy::pedantic)]

use num_traits::{Bounded, CheckedSub, One, PrimInt, Zero};

/// Our own custom error enumeration, to provide better errors than
/// [`std::num::ParseIntError`].
#[derive(Debug, Clone, PartialEq)]
pub enum IntegerParseError {
    /// The radix we received was invalid (outside of [2,36])
    InvalidRadix,
    /// There was no data to parse.
    Empty,
    /// The value we got was negative but we're parsing an unsigned number.
    /// Note: `-0` is at the moment considered to be an error when parsing unsigned numbers.
    UnsignedNegative,
    /// The character was not valid in our current base
    InvalidCharacter(char),
    // TODO: It would be nice to completely avoid the need for this error since it means that
    // we can't parse a u4 in base 17. Not major, since that isn't a builtin integer type
    // and obscure case, but would be nice to support.
    /// The radix, when converted to T, would exceed the bounds of the type.
    /// This shouldn't happen for normal integer types, since they can all hold a value in
    /// [2, 36].
    RadixExceedsBounds,
    /// The digit would exceed the bounds of the type itself. Impressive.
    /// Probably occurs if you try doing ex: base 36 with digit Z on a `u4`.
    DigitExceedsBounds,
    /// Failed to change the sign. This is a weird error to get, and it likely shouldn't happen
    /// with normal integer types.
    SignModifyFailure,
    /// When adding the new digit, it would overflow the value. Thus the digits produce a number
    /// out of bounds of T.
    ExceedsBounds,
    // /// When subtracting the new digit, it would underflow the value.
    // NegativeOverflow,
}
impl std::fmt::Display for IntegerParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRadix => f.write_str("radix was outside of [2,36]"),
            Self::Empty => f.write_str("no data"),
            Self::UnsignedNegative => f.write_str("expected unsigned number to not be negative"),
            Self::InvalidCharacter(ch) => f.write_fmt(format_args!("invalid character '{}'", ch)),
            Self::RadixExceedsBounds => f.write_str("radix exceeds bounds of type"),
            Self::DigitExceedsBounds => f.write_str("digit exceeds bounds of type"),
            Self::SignModifyFailure => f.write_str("failed to change sign to negative"),
            Self::ExceedsBounds => f.write_str("value exceeds bounds of type"),
        }
    }
}

#[derive(PartialEq)]
enum NumberSign {
    Positive,
    Negative,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum NumberBase {
    /// 2 0b
    Binary,
    /// 8 0o
    Octal,
    /// 10 0d
    Decimal,
    /// 16 0x
    Hex,
    /// Other radices. Should only be in [2,36]
    /// Note: The base being an existing named variant is supported.
    Other(u32),
}
impl NumberBase {
    /// Returns the base from the given prefix.
    /// Returns None if it was not reocgnized.
    #[must_use]
    pub fn prefix_for(prefix: char) -> Option<Self> {
        Some(match prefix {
            'b' => NumberBase::Binary,
            'o' => NumberBase::Octal,
            'h' => NumberBase::Decimal,
            'x' => NumberBase::Hex,
            _ => return None,
        })
    }

    /// Check if the character is a prefix for this specific base
    #[must_use]
    pub fn is_prefix(self, ch: char) -> bool {
        Self::prefix_for(ch) == Some(self)
    }

    /// Get the radix/base integer.
    #[must_use]
    pub fn as_radix(self) -> u32 {
        match self {
            Self::Binary => 2,
            Self::Octal => 8,
            Self::Decimal => 10,
            Self::Hex => 16,
            Self::Other(radix) => radix,
        }
    }

    /// Checks if the character is a valid digit in the base.
    #[must_use]
    pub fn is_digit(self, c: char) -> bool {
        c.is_digit(self.as_radix())
    }

    /// Convert the character to its value in the base
    /// (if the character is valid in the base)
    #[must_use]
    pub fn to_value(self, c: char) -> Option<u32> {
        c.to_digit(self.as_radix())
    }

    /// Tells whether the radix is (generally) valid.
    /// Aka: in the range [2,36]
    #[must_use]
    pub fn is_allowed_radix(self) -> bool {
        (2..=36).contains(&self.as_radix())
    }
}

/// debug_assert but it performs the left-hand side.
macro_rules! debug_assert_eq_do {
    ($left:expr, $right:expr) => {{
        let v = $left;
        debug_assert_eq!(v, $right);
    }};
}

/// Parse an integer from a `&str`
/// ```
/// # use prog_integer_parse::{parse_integer, NumberBase};
/// let text = "409";
/// let value: u16 = parse_integer::<u16>(text, NumberBase::Decimal).unwrap();
/// assert_eq!(value, 409);
/// let value: u16 = parse_integer::<u16>(text, NumberBase::Hex).unwrap();
/// assert_eq!(value, 1033);
/// let text = "0x4a";
/// let value: u16 = parse_integer::<u16>(text, NumberBase::Octal).unwrap();
/// assert_eq!(value, 0x4a); // Prefix overrides default base
/// ```
/// # Errors
/// Same reasons as [`parse_integer_iter_raw`].
/// # Panics
/// Same reasons as [`parse_integer_iter_raw`]
pub fn parse_integer<T: PrimInt>(
    text: &str,
    default_radix: NumberBase,
) -> Result<T, IntegerParseError> {
    parse_integer_iter::<T, _>(text.chars(), default_radix)
}
// TODO: Should we require the base to be an enumeration? Better type safety..
/// Parse an integer from an iterator.
/// ```
/// # use prog_integer_parse::{parse_integer_iter, NumberBase};
/// let text = "674"; // 476.
/// // Reminder: This assumes that the left-most digit is the lowest value digit.
/// let value: u16 = parse_integer_iter::<u16, _>(text.chars(), NumberBase::Decimal).unwrap();
/// assert_eq!(value, 674);
/// let value: u16 = parse_integer_iter::<u16, _>(text.chars(), NumberBase::Octal).unwrap();
/// assert_eq!(value, 0o674); // 0o476
/// ```
/// # Errors
/// - If radix is out of bounds [2, 36]
/// - No digits
/// - Invalid sign for type (negative for unsigned types)
/// - Overflows
/// # Panics
/// Does *not* panic when `radix` is outside of [2, 36] (which `std` parsing does). Returns error.
/// - Given text iterator panics
pub fn parse_integer_iter<T: PrimInt, I: Iterator<Item = char>>(
    text: I,
    default_radix: NumberBase,
) -> Result<T, IntegerParseError> {
    // std `T::from_str_radix` only supports radices in [2, 36] and we imitate that.
    // We error even if we aren't using the radix to avoid weird errors that only appear
    // later in code (ex, often parsing hex but then later a decimal without a prefix)
    if !default_radix.is_allowed_radix() {
        return Err(IntegerParseError::InvalidRadix);
    }

    // Fuse the iterator to ensure we receive None again.
    let text = text.fuse();
    // We need to be able to peek at least once for some parsing.
    let mut text = text.peekable();

    if text.peek().is_none() {
        return Err(IntegerParseError::Empty);
    }

    let sign: NumberSign = match text.peek().copied() {
        Some('-') => {
            debug_assert_eq_do!(text.next(), Some('-'));
            NumberSign::Negative
        }
        Some('+') => {
            debug_assert_eq_do!(text.next(), Some('+'));
            NumberSign::Positive
        }
        // Default sign.
        _ => NumberSign::Positive,
    };

    if text.peek().is_none() {
        return Err(IntegerParseError::Empty);
    }

    let sign_dir = match sign {
        NumberSign::Positive => T::one(),
        NumberSign::Negative => {
            get_negative_one::<T>().ok_or(IntegerParseError::UnsignedNegative)?
        }
    };

    let radix: NumberBase = if text.peek().copied() == Some('0') {
        // Note: Essentially we ignore the beginning 0 to check the prefix.
        debug_assert_eq_do!(text.next(), Some('0'));

        let prefix_char = text.peek().copied();
        if let Some(radix) = prefix_char.and_then(NumberBase::prefix_for) {
            debug_assert_eq_do!(text.next(), prefix_char);

            if text.peek().is_none() {
                return Err(IntegerParseError::Empty);
            }

            radix
        } else {
            // Non valid prefix character. Might also just be a digit.
            // TODO: Optional support for digits which would imply that it is octal?
            default_radix
        }
    } else {
        default_radix
    };

    // This should always be true.
    debug_assert!(radix.is_allowed_radix());

    let radix_t: T =
        num_traits::cast(radix.as_radix()).ok_or(IntegerParseError::RadixExceedsBounds)?;

    let mut result: T = T::zero();

    for ch in text {
        // Panic-Avoidance: `radix` is already checked to be within [2, 36] range.
        // Get the value of the current digit (if it is a digit)
        if let Some(digit_value) = radix.to_value(ch) {
            // Convert digit into type T. If the digit can't fit then it certainly can't fit
            // when we later multiply it to put it in the proper place.
            let digit_value: T =
                num_traits::cast(digit_value).ok_or(IntegerParseError::DigitExceedsBounds)?;
            // Swap the sign of the digit if we need to.
            let digit_value: T = digit_value
                .checked_mul(&sign_dir)
                .ok_or(IntegerParseError::SignModifyFailure)?;
            result = result
                .checked_mul(&radix_t)
                .ok_or(IntegerParseError::ExceedsBounds)?;
            result = result
                .checked_add(&digit_value)
                .ok_or(IntegerParseError::ExceedsBounds)?;
        } else {
            // TODO: Support _ separators?
            // It wasn't a digit in our base, and we currently don't support anything else.
            return Err(IntegerParseError::InvalidCharacter(ch));
        }
    }

    Ok(result)
}

/// Get if the generic integer type can be negative
/// Because for _some_ reason, num-traits doesn't have it.
#[must_use]
fn is_signed_type<T: Bounded + Zero + PartialOrd>() -> bool {
    // TODO: Should we check to make sure that it also has positive values?
    T::min_value() < T::zero()
}

/// Get the value of negative one from the type if it exists.
/// Assumes that it is 0 - 1.
#[must_use]
fn get_negative_one<T: Bounded + Zero + One + CheckedSub + PartialOrd>() -> Option<T> {
    if is_signed_type::<T>() {
        T::zero().checked_sub(&T::one())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use crate::{parse_integer, IntegerParseError, NumberBase};

    // TODO: Could automate some of these tests if I could use from_digit but that is currently
    // nightly only

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_leading_1_overflows() {
        // Absurdly large loop that checks for basic overflows essentially at each radix
        for radix in 2..=36 {
            let base = NumberBase::Other(radix);
            assert_eq!(parse_integer::<u8>("0", base), Ok(0));
            assert_eq!(parse_integer::<u8>("1", base), Ok(1));
            assert_eq!(parse_integer::<u8>("000000000", base), Ok(0));
            assert_eq!(parse_integer::<u32>("10", base), Ok(radix));
            assert_eq!(parse_integer::<u32>("100", base), Ok(radix.pow(2)));
            assert_eq!(parse_integer::<u32>("1000", base), Ok(radix.pow(3)));
            assert_eq!(parse_integer::<u32>("10000", base), Ok(radix.pow(4)));
            assert_eq!(parse_integer::<u32>("100000", base), Ok(radix.pow(5)));
            assert_eq!(parse_integer::<u32>("1000000", base), Ok(radix.pow(6)));
            // radixes greater than 23 start overflowing in a u32 for the radix
            let value = parse_integer::<u32>("10000000", base);
            if radix <= 23 {
                assert_eq!(value, Ok(radix.pow(7)));
                // radixes greater than 15 start overflowing
                let value = parse_integer::<u32>("100000000", base);
                if radix <= 15 {
                    assert_eq!(value, Ok(radix.pow(8)));
                    // radixes greater than 11 start overflowing
                    let value = parse_integer::<u32>("1000000000", base);
                    if radix <= 11 {
                        assert_eq!(value, Ok(radix.pow(9)));
                        let value = parse_integer::<u32>("10000000000", base);
                        if radix <= 9 {
                            assert_eq!(value, Ok(radix.pow(10)));
                            let value = parse_integer::<u32>("100000000000", base);
                            if radix <= 7 {
                                assert_eq!(value, Ok(radix.pow(11)));
                                let value = parse_integer::<u32>("1000000000000", base);
                                if radix <= 6 {
                                    assert_eq!(value, Ok(radix.pow(12)));
                                    let value = parse_integer::<u32>("10000000000000", base);
                                    if radix <= 5 {
                                        assert_eq!(value, Ok(radix.pow(13)));
                                        let value = parse_integer::<u32>("100000000000000", base);
                                        if radix <= 4 {
                                            assert_eq!(value, Ok(radix.pow(14)));
                                            let value =
                                                parse_integer::<u32>("1000000000000000", base);
                                            assert_eq!(value, Ok(radix.pow(15)));
                                            let value =
                                                parse_integer::<u32>("10000000000000000", base);
                                            if radix <= 3 {
                                                assert_eq!(value, Ok(radix.pow(16)));
                                                assert_eq!(
                                                    parse_integer::<u32>(
                                                        "100000000000000000",
                                                        base
                                                    ),
                                                    Ok(radix.pow(17))
                                                );
                                                assert_eq!(
                                                    parse_integer::<u32>(
                                                        "1000000000000000000",
                                                        base
                                                    ),
                                                    Ok(radix.pow(18))
                                                );
                                                assert_eq!(
                                                    parse_integer::<u32>(
                                                        "10000000000000000000",
                                                        base
                                                    ),
                                                    Ok(radix.pow(19))
                                                );
                                                assert_eq!(
                                                    parse_integer::<u32>(
                                                        "100000000000000000000",
                                                        base
                                                    ),
                                                    Ok(radix.pow(20))
                                                );
                                                // We don't check radix-2 here.
                                            } else {
                                                assert_eq!(
                                                    value,
                                                    Err(IntegerParseError::ExceedsBounds)
                                                );
                                            }
                                        } else {
                                            assert_eq!(
                                                value,
                                                Err(IntegerParseError::ExceedsBounds)
                                            );
                                        }
                                    } else {
                                        assert_eq!(value, Err(IntegerParseError::ExceedsBounds));
                                    }
                                } else {
                                    assert_eq!(value, Err(IntegerParseError::ExceedsBounds));
                                }
                            } else {
                                assert_eq!(value, Err(IntegerParseError::ExceedsBounds));
                            }
                        } else {
                            assert_eq!(value, Err(IntegerParseError::ExceedsBounds));
                        }
                    } else {
                        assert_eq!(value, Err(IntegerParseError::ExceedsBounds));
                    }
                } else {
                    assert_eq!(value, Err(IntegerParseError::ExceedsBounds));
                }
            } else {
                assert_eq!(value, Err(IntegerParseError::ExceedsBounds));
            }
        }
    }

    #[test]
    fn test_basic_negative() {
        for radix in 2..=36 {
            let base = NumberBase::Other(radix);
            assert_eq!(parse_integer("-1", base), Ok(-1_i32));
        }
    }

    #[test]
    fn test_basic_prefixes() {
        for radix in 2..=36 {
            let base = NumberBase::Other(radix);
            // Zeroes
            assert_eq!(parse_integer("0b0", base), Ok(0));
            assert_eq!(parse_integer("0o0", base), Ok(0));
            assert_eq!(parse_integer("0h0", base), Ok(0));
            assert_eq!(parse_integer("0x0", base), Ok(0));

            // Several zeroes
            assert_eq!(
                parse_integer(
                    "0b0000000000000000000000000000000000000000000000000000000",
                    base
                ),
                Ok(0)
            );
            assert_eq!(
                parse_integer(
                    "0o0000000000000000000000000000000000000000000000000000000",
                    base
                ),
                Ok(0)
            );
            assert_eq!(
                parse_integer(
                    "0h0000000000000000000000000000000000000000000000000000000",
                    base
                ),
                Ok(0)
            );
            assert_eq!(
                parse_integer(
                    "0x0000000000000000000000000000000000000000000000000000000",
                    base
                ),
                Ok(0)
            );

            // Single leading 1
            assert_eq!(parse_integer("0b10", base), Ok(2));
            assert_eq!(parse_integer("0o10", base), Ok(8));
            assert_eq!(parse_integer("0h10", base), Ok(10));
            assert_eq!(parse_integer("0x10", base), Ok(16));

            assert_eq!(parse_integer("0b11", base), Ok(3));
            assert_eq!(parse_integer("0o11", base), Ok(9));
            assert_eq!(parse_integer("0h11", base), Ok(11));
            assert_eq!(parse_integer("0x11", base), Ok(17));

            assert_eq!(parse_integer("0o12", base), Ok(10));
            assert_eq!(parse_integer("0h12", base), Ok(12));
            assert_eq!(parse_integer("0x12", base), Ok(18));

            assert_eq!(parse_integer("0o13", base), Ok(11));
            assert_eq!(parse_integer("0h13", base), Ok(13));
            assert_eq!(parse_integer("0x13", base), Ok(19));

            assert_eq!(parse_integer("0o14", base), Ok(12));
            assert_eq!(parse_integer("0h14", base), Ok(14));
            assert_eq!(parse_integer("0x14", base), Ok(20));

            assert_eq!(parse_integer("0o15", base), Ok(13));
            assert_eq!(parse_integer("0h15", base), Ok(15));
            assert_eq!(parse_integer("0x15", base), Ok(21));
        }
    }

    #[test]
    fn maximal_minimal_tests() {
        assert_eq!(parse_integer("255", NumberBase::Decimal), Ok(u8::MAX));
        assert_eq!(parse_integer("127", NumberBase::Decimal), Ok(i8::MAX));
        assert_eq!(parse_integer("-128", NumberBase::Decimal), Ok(i8::MIN));
        assert_eq!(parse_integer("65535", NumberBase::Decimal), Ok(u16::MAX));
        assert_eq!(parse_integer("32767", NumberBase::Decimal), Ok(i16::MAX));
        assert_eq!(parse_integer("-32768", NumberBase::Decimal), Ok(i16::MIN));

        assert_eq!(parse_integer("0xff", NumberBase::Decimal), Ok(u8::MAX));
        assert_eq!(parse_integer("0x7f", NumberBase::Decimal), Ok(i8::MAX));
        assert_eq!(parse_integer("-0x80", NumberBase::Decimal), Ok(i8::MIN));
        assert_eq!(parse_integer("0xffff", NumberBase::Decimal), Ok(u16::MAX));
        assert_eq!(parse_integer("0x7fff", NumberBase::Decimal), Ok(i16::MAX));
        assert_eq!(parse_integer("-0x8000", NumberBase::Decimal), Ok(i16::MIN));
    }

    #[test]
    fn more_complex_negatives() {
        assert_eq!(parse_integer("-92", NumberBase::Decimal), Ok(-92));
        for radix in 2..36 {
            let base = NumberBase::Other(radix);
            assert_eq!(parse_integer("-0b10", base), Ok(-2));
            assert_eq!(parse_integer("-0o13", base), Ok(-11));
            assert_eq!(parse_integer("-0h13", base), Ok(-13));
            assert_eq!(parse_integer("-0x13", base), Ok(-19));
        }
    }

    #[test]
    fn negative_unsigned() {
        assert_eq!(
            parse_integer::<u8>("-0", NumberBase::Decimal),
            Err(IntegerParseError::UnsignedNegative)
        );
        assert_eq!(
            parse_integer::<u8>("-1", NumberBase::Decimal),
            Err(IntegerParseError::UnsignedNegative)
        );
        assert_eq!(
            parse_integer::<u8>("-255", NumberBase::Decimal),
            Err(IntegerParseError::UnsignedNegative)
        );
        assert_eq!(
            parse_integer::<u8>("-0x0", NumberBase::Decimal),
            Err(IntegerParseError::UnsignedNegative)
        );
        assert_eq!(
            parse_integer::<u8>("-0xfa", NumberBase::Decimal),
            Err(IntegerParseError::UnsignedNegative)
        );
    }

    #[test]
    fn empty_tests() {
        assert_eq!(
            parse_integer::<u8>("", NumberBase::Decimal),
            Err(IntegerParseError::Empty)
        );
        assert_eq!(
            parse_integer::<u8>("0b", NumberBase::Decimal),
            Err(IntegerParseError::Empty)
        );
        assert_eq!(
            parse_integer::<u8>("0o", NumberBase::Decimal),
            Err(IntegerParseError::Empty)
        );
        assert_eq!(
            parse_integer::<u8>("0h", NumberBase::Decimal),
            Err(IntegerParseError::Empty)
        );
        assert_eq!(
            parse_integer::<u8>("0x", NumberBase::Decimal),
            Err(IntegerParseError::Empty)
        );
        assert_eq!(
            parse_integer::<i8>("-0x", NumberBase::Decimal),
            Err(IntegerParseError::Empty)
        );
        assert_eq!(
            parse_integer::<i8>("-", NumberBase::Decimal),
            Err(IntegerParseError::Empty)
        );
        assert_eq!(
            parse_integer::<i8>("+", NumberBase::Decimal),
            Err(IntegerParseError::Empty)
        );
    }

    #[test]
    fn unexpected_character() {
        assert_eq!(
            parse_integer::<u8>(" ", NumberBase::Decimal),
            Err(IntegerParseError::InvalidCharacter(' '))
        );
        assert_eq!(
            parse_integer::<u8>("       \t \n", NumberBase::Decimal),
            Err(IntegerParseError::InvalidCharacter(' '))
        );
        assert_eq!(
            parse_integer::<u8>("\n", NumberBase::Decimal),
            Err(IntegerParseError::InvalidCharacter('\n'))
        );
        assert_eq!(
            parse_integer::<u8>("\t", NumberBase::Decimal),
            Err(IntegerParseError::InvalidCharacter('\t'))
        );
        assert_eq!(
            parse_integer::<u8>("a", NumberBase::Decimal),
            Err(IntegerParseError::InvalidCharacter('a'))
        );
        assert_eq!(
            parse_integer::<u8>("z", NumberBase::Decimal),
            Err(IntegerParseError::InvalidCharacter('z'))
        );
        assert_eq!(
            parse_integer::<u8>("2", NumberBase::Binary),
            Err(IntegerParseError::InvalidCharacter('2'))
        );
        assert_eq!(
            parse_integer::<u8>("0b2", NumberBase::Decimal),
            Err(IntegerParseError::InvalidCharacter('2'))
        );
        assert_eq!(
            parse_integer::<u8>("0o19", NumberBase::Decimal),
            Err(IntegerParseError::InvalidCharacter('9'))
        );
    }
}
