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
// TODO: Implement display.

#[derive(PartialEq)]
enum NumberSign {
    Positive,
    Negative,
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
/// # use prog_integer_parse::parse_integer;
/// let text = "409";
/// let value: u16 = parse_integer::<u16>(text, 10).unwrap();
/// assert_eq!(value, 409);
/// let value: u16 = parse_integer::<u16>(text, 16).unwrap();
/// assert_eq!(value, 1033);
/// let text = "0x4a";
/// let value: u16 = parse_integer::<u16>(text, 8).unwrap();
/// assert_eq!(value, 0x4a); // Prefix overrides default base
/// ```
/// # Errors
/// Same reasons as [`parse_integer_iter_raw`].
/// # Panics
/// Same reasons as [`parse_integer_iter_raw`]
pub fn parse_integer<T: PrimInt>(text: &str, default_radix: u32) -> Result<T, IntegerParseError> {
    parse_integer_iter::<T, _>(text.chars(), default_radix)
}
// TODO: Should we require the base to be an enumeration? Better type safety..
/// Parse an integer from an iterator.
/// ```
/// # use prog_integer_parse::parse_integer_iter;
/// let text = "674"; // 476.
/// // Reminder: This assumes that the left-most digit is the lowest value digit.
/// let value: u16 = parse_integer_iter::<u16, _>(text.chars(), 10).unwrap();
/// assert_eq!(value, 674);
/// let value: u16 = parse_integer_iter::<u16, _>(text.chars(), 8).unwrap();
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
    default_radix: u32,
) -> Result<T, IntegerParseError> {
    // std `T::from_str_radix` only supports radices in [2, 36] and we imitate that.
    // We error even if we aren't using the radix to avoid weird errors that only appear
    // later in code (ex, often parsing hex but then later a decimal without a prefix)
    if !is_allowed_radix(default_radix) {
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
            // println!("Negative sign");
            debug_assert_eq_do!(text.next(), Some('-'));
            NumberSign::Negative
        }
        Some('+') => {
            // println!("Positive sign");
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

    // println!("Value: {:?}", text.peek().copied());
    let radix: u32 = if text.peek().copied() == Some('0') {
        // Note: Essentially we ignore the beginning 0 to check the prefix.
        debug_assert_eq_do!(text.next(), Some('0'));

        let prefix_char = text.peek().copied();
        // println!("Prefix: {:?}", prefix_char);
        if let Some(radix) = prefix_char.and_then(get_radix) {
            // println!("Radix prefix");
            debug_assert_eq_do!(text.next(), prefix_char);

            if text.peek().is_none() {
                return Err(IntegerParseError::Empty);
            }

            radix
        } else {
            // Non valid prefix character. Might also just be a digit.
            // TODO: Optional support for digits which would imply that it is octal?
            // println!("No radix prefix");
            default_radix
        }
    } else {
        default_radix
    };

    // println!("Radix: {}", radix);

    // This should always be true.
    debug_assert!(is_allowed_radix(radix));

    let radix_t: T = num_traits::cast(radix).ok_or(IntegerParseError::RadixExceedsBounds)?;

    let mut result: T = T::zero();

    // TODO: Make negative on first value pass through
    // because we can't store (ex) 128 in an i8 then make it negative
    // let mut index: T = T::one();
    for ch in text {
        // Panic-Avoidance: `radix` is already checked to be within [2, 36] range.
        // Get the value of the current digit (if it is a digit)
        if let Some(digit_value) = ch.to_digit(radix) {
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

#[must_use]
fn is_allowed_radix(radix: u32) -> bool {
    (2..=36).contains(&radix)
}

/// Get the radix that the character implies.
/// Returns None or a radix within the range [2, 36]
#[must_use]
fn get_radix(ch: char) -> Option<u32> {
    Some(match ch {
        'b' => 2,
        'o' => 8,
        'h' => 10,
        'x' => 16,
        _ => return None,
    })
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
    use crate::{parse_integer, IntegerParseError};

    // TODO: Could automate some of these tests if I could use from_digit but that is currently
    // nightly only

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_leading_1_overflows() {
        // Absurdly large loop that checks for basic overflows essentially at each radix
        for radix in 2..=36 {
            assert_eq!(parse_integer::<u8>("0", radix), Ok(0));
            assert_eq!(parse_integer::<u8>("1", radix), Ok(1));
            assert_eq!(parse_integer::<u8>("000000000", radix), Ok(0));
            assert_eq!(parse_integer::<u32>("10", radix), Ok(radix));
            assert_eq!(parse_integer::<u32>("100", radix), Ok(radix.pow(2)));
            assert_eq!(parse_integer::<u32>("1000", radix), Ok(radix.pow(3)));
            assert_eq!(parse_integer::<u32>("10000", radix), Ok(radix.pow(4)));
            assert_eq!(parse_integer::<u32>("100000", radix), Ok(radix.pow(5)));
            assert_eq!(parse_integer::<u32>("1000000", radix), Ok(radix.pow(6)));
            // radixes greater than 23 start overflowing in a u32 for the radix
            let value = parse_integer::<u32>("10000000", radix);
            if radix <= 23 {
                assert_eq!(value, Ok(radix.pow(7)));
                // radixes greater than 15 start overflowing
                let value = parse_integer::<u32>("100000000", radix);
                if radix <= 15 {
                    assert_eq!(value, Ok(radix.pow(8)));
                    // radixes greater than 11 start overflowing
                    let value = parse_integer::<u32>("1000000000", radix);
                    if radix <= 11 {
                        assert_eq!(value, Ok(radix.pow(9)));
                        let value = parse_integer::<u32>("10000000000", radix);
                        if radix <= 9 {
                            assert_eq!(value, Ok(radix.pow(10)));
                            let value = parse_integer::<u32>("100000000000", radix);
                            if radix <= 7 {
                                println!("Radix: {}", radix);
                                assert_eq!(value, Ok(radix.pow(11)));
                                let value = parse_integer::<u32>("1000000000000", radix);
                                if radix <= 6 {
                                    assert_eq!(value, Ok(radix.pow(12)));
                                    let value = parse_integer::<u32>("10000000000000", radix);
                                    if radix <= 5 {
                                        assert_eq!(value, Ok(radix.pow(13)));
                                        let value = parse_integer::<u32>("100000000000000", radix);
                                        if radix <= 4 {
                                            assert_eq!(value, Ok(radix.pow(14)));
                                            let value =
                                                parse_integer::<u32>("1000000000000000", radix);
                                            assert_eq!(value, Ok(radix.pow(15)));
                                            let value =
                                                parse_integer::<u32>("10000000000000000", radix);
                                            if radix <= 3 {
                                                assert_eq!(value, Ok(radix.pow(16)));
                                                assert_eq!(
                                                    parse_integer::<u32>(
                                                        "100000000000000000",
                                                        radix
                                                    ),
                                                    Ok(radix.pow(17))
                                                );
                                                assert_eq!(
                                                    parse_integer::<u32>(
                                                        "1000000000000000000",
                                                        radix
                                                    ),
                                                    Ok(radix.pow(18))
                                                );
                                                assert_eq!(
                                                    parse_integer::<u32>(
                                                        "10000000000000000000",
                                                        radix
                                                    ),
                                                    Ok(radix.pow(19))
                                                );
                                                assert_eq!(
                                                    parse_integer::<u32>(
                                                        "100000000000000000000",
                                                        radix
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
            assert_eq!(parse_integer("-1", radix), Ok(-1_i32));
        }
    }

    #[test]
    fn test_basic_prefixes() {
        for radix in 2..=36 {
            // Zeroes
            assert_eq!(parse_integer("0b0", radix), Ok(0));
            assert_eq!(parse_integer("0o0", radix), Ok(0));
            assert_eq!(parse_integer("0h0", radix), Ok(0));
            assert_eq!(parse_integer("0x0", radix), Ok(0));

            // Several zeroes
            assert_eq!(
                parse_integer(
                    "0b0000000000000000000000000000000000000000000000000000000",
                    radix
                ),
                Ok(0)
            );
            assert_eq!(
                parse_integer(
                    "0o0000000000000000000000000000000000000000000000000000000",
                    radix
                ),
                Ok(0)
            );
            assert_eq!(
                parse_integer(
                    "0h0000000000000000000000000000000000000000000000000000000",
                    radix
                ),
                Ok(0)
            );
            assert_eq!(
                parse_integer(
                    "0x0000000000000000000000000000000000000000000000000000000",
                    radix
                ),
                Ok(0)
            );

            // Single leading 1
            assert_eq!(parse_integer("0b10", radix), Ok(2));
            assert_eq!(parse_integer("0o10", radix), Ok(8));
            assert_eq!(parse_integer("0h10", radix), Ok(10));
            assert_eq!(parse_integer("0x10", radix), Ok(16));

            assert_eq!(parse_integer("0b11", radix), Ok(3));
            assert_eq!(parse_integer("0o11", radix), Ok(9));
            assert_eq!(parse_integer("0h11", radix), Ok(11));
            assert_eq!(parse_integer("0x11", radix), Ok(17));

            assert_eq!(parse_integer("0o12", radix), Ok(10));
            assert_eq!(parse_integer("0h12", radix), Ok(12));
            assert_eq!(parse_integer("0x12", radix), Ok(18));

            assert_eq!(parse_integer("0o13", radix), Ok(11));
            assert_eq!(parse_integer("0h13", radix), Ok(13));
            assert_eq!(parse_integer("0x13", radix), Ok(19));

            assert_eq!(parse_integer("0o14", radix), Ok(12));
            assert_eq!(parse_integer("0h14", radix), Ok(14));
            assert_eq!(parse_integer("0x14", radix), Ok(20));

            assert_eq!(parse_integer("0o15", radix), Ok(13));
            assert_eq!(parse_integer("0h15", radix), Ok(15));
            assert_eq!(parse_integer("0x15", radix), Ok(21));
        }
    }

    #[test]
    fn maximal_minimal_tests() {
        assert_eq!(parse_integer("255", 10), Ok(u8::MAX));
        assert_eq!(parse_integer("127", 10), Ok(i8::MAX));
        assert_eq!(parse_integer("-128", 10), Ok(i8::MIN));
        assert_eq!(parse_integer("65535", 10), Ok(u16::MAX));
        assert_eq!(parse_integer("32767", 10), Ok(i16::MAX));
        assert_eq!(parse_integer("-32768", 10), Ok(i16::MIN));

        assert_eq!(parse_integer("0xff", 10), Ok(u8::MAX));
        assert_eq!(parse_integer("0x7f", 10), Ok(i8::MAX));
        assert_eq!(parse_integer("-0x80", 10), Ok(i8::MIN));
        assert_eq!(parse_integer("0xffff", 10), Ok(u16::MAX));
        assert_eq!(parse_integer("0x7fff", 10), Ok(i16::MAX));
        assert_eq!(parse_integer("-0x8000", 10), Ok(i16::MIN));
    }

    #[test]
    fn more_complex_negatives() {
        assert_eq!(parse_integer("-92", 10), Ok(-92));
        for radix in 2..36 {
            assert_eq!(parse_integer("-0b10", radix), Ok(-2));
            assert_eq!(parse_integer("-0o13", radix), Ok(-11));
            assert_eq!(parse_integer("-0h13", radix), Ok(-13));
            assert_eq!(parse_integer("-0x13", radix), Ok(-19));
        }
    }

    #[test]
    fn negative_unsigned() {
        assert_eq!(
            parse_integer::<u8>("-0", 10),
            Err(IntegerParseError::UnsignedNegative)
        );
        assert_eq!(
            parse_integer::<u8>("-1", 10),
            Err(IntegerParseError::UnsignedNegative)
        );
        assert_eq!(
            parse_integer::<u8>("-255", 10),
            Err(IntegerParseError::UnsignedNegative)
        );
        assert_eq!(
            parse_integer::<u8>("-0x0", 10),
            Err(IntegerParseError::UnsignedNegative)
        );
        assert_eq!(
            parse_integer::<u8>("-0xfa", 10),
            Err(IntegerParseError::UnsignedNegative)
        );
    }

    #[test]
    fn empty_tests() {
        assert_eq!(parse_integer::<u8>("", 10), Err(IntegerParseError::Empty));
        assert_eq!(parse_integer::<u8>("0b", 10), Err(IntegerParseError::Empty));
        assert_eq!(parse_integer::<u8>("0o", 10), Err(IntegerParseError::Empty));
        assert_eq!(parse_integer::<u8>("0h", 10), Err(IntegerParseError::Empty));
        assert_eq!(parse_integer::<u8>("0x", 10), Err(IntegerParseError::Empty));
        assert_eq!(
            parse_integer::<i8>("-0x", 10),
            Err(IntegerParseError::Empty)
        );
        assert_eq!(parse_integer::<i8>("-", 10), Err(IntegerParseError::Empty));
        assert_eq!(parse_integer::<i8>("+", 10), Err(IntegerParseError::Empty));
    }

    #[test]
    fn unexpected_character() {
        assert_eq!(
            parse_integer::<u8>(" ", 10),
            Err(IntegerParseError::InvalidCharacter(' '))
        );
        assert_eq!(
            parse_integer::<u8>("       \t \n", 10),
            Err(IntegerParseError::InvalidCharacter(' '))
        );
        assert_eq!(
            parse_integer::<u8>("\n", 10),
            Err(IntegerParseError::InvalidCharacter('\n'))
        );
        assert_eq!(
            parse_integer::<u8>("\t", 10),
            Err(IntegerParseError::InvalidCharacter('\t'))
        );
        assert_eq!(
            parse_integer::<u8>("a", 10),
            Err(IntegerParseError::InvalidCharacter('a'))
        );
        assert_eq!(
            parse_integer::<u8>("z", 10),
            Err(IntegerParseError::InvalidCharacter('z'))
        );
        assert_eq!(
            parse_integer::<u8>("2", 2),
            Err(IntegerParseError::InvalidCharacter('2'))
        );
        assert_eq!(
            parse_integer::<u8>("0b2", 10),
            Err(IntegerParseError::InvalidCharacter('2'))
        );
        assert_eq!(
            parse_integer::<u8>("0o19", 10),
            Err(IntegerParseError::InvalidCharacter('9'))
        );
    }
}
