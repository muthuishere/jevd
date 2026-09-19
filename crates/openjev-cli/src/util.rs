//! Small shared things with no home of their own.

use std::io::IsTerminal;
use std::time::{SystemTime, UNIX_EPOCH};

/// TTY governs colour, spinners and progress — never the shape of stdout (design 02
/// rule 4). stderr and stdin are asked about separately: progress goes to stderr and the
/// consent prompt needs a readable stdin, and the two are redirected independently.
pub fn stderr_is_tty() -> bool {
    std::io::stderr().is_terminal()
}

pub fn stdin_is_tty() -> bool {
    std::io::stdin().is_terminal()
}

/// RFC 3339 in UTC, second resolution. Hand-rolled rather than a date crate: we format
/// timestamps and never parse or do arithmetic on them, and a civil-date conversion is
/// 20 lines that cannot surprise us at a version bump.
pub fn rfc3339_now() -> String {
    rfc3339(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    )
}

pub fn rfc3339(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let tod = secs % 86_400;
    let (y, m, d) = civil_from_days(days);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        tod / 3600,
        (tod % 3600) / 60,
        tod % 60
    )
}

/// Howard Hinnant's days-from-civil, inverted. Public-domain algorithm, exact for all
/// dates we can represent.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// 128 bits of randomness, hex. Enough to correlate a log line with a bug report and
/// nothing more — it is not a secret and is echoed to the client on purpose.
pub fn request_id() -> String {
    use rand::Rng;
    let mut b = [0u8; 16];
    rand::rng().fill(&mut b);
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// A fresh bearer token. Printed once, stored hashed.
pub fn generate_token() -> String {
    use rand::Rng;
    let mut b = [0u8; 32];
    rand::rng().fill(&mut b);
    b.iter().map(|x| format!("{x:02x}")).collect()
}

pub fn sha256_hex(s: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    h.finalize().iter().map(|x| format!("{x:02x}")).collect()
}

/// Constant-time bearer comparison over the hashes, so a timing side channel cannot walk
/// the token out of us one byte at a time.
pub fn token_matches(presented: &str, expected_hash: &str) -> bool {
    use subtle::ConstantTimeEq;
    let got = sha256_hex(presented);
    got.as_bytes().ct_eq(expected_hash.as_bytes()).into()
}

pub fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1024.0 && i + 1 < UNITS.len() {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{n} B")
    } else {
        format!("{v:.1} {}", UNITS[i])
    }
}

pub fn human_secs(s: f64) -> String {
    if s < 60.0 {
        format!("{s:.1}s")
    } else {
        format!("{}m{:02}s", (s / 60.0) as u64, (s % 60.0) as u64)
    }
}

/// An address only this machine can reach. The whole auth default hangs off this
/// predicate, so it is a function with tests, not an inline `starts_with("127.")`.
pub fn is_loopback(host: &str) -> bool {
    match host.parse::<std::net::IpAddr>() {
        Ok(ip) => ip.is_loopback(),
        Err(_) => host.eq_ignore_ascii_case("localhost"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_and_a_known_date_format_exactly() {
        assert_eq!(rfc3339(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339(1_758_268_442), "2025-09-19T07:54:02Z");
    }

    #[test]
    fn loopback_is_every_loopback_and_nothing_else() {
        assert!(is_loopback("127.0.0.1"));
        assert!(is_loopback("127.1.2.3"));
        assert!(is_loopback("::1"));
        assert!(is_loopback("localhost"));
        assert!(!is_loopback("0.0.0.0"));
        assert!(!is_loopback("192.168.1.10"));
        assert!(!is_loopback("::"));
    }

    #[test]
    fn tokens_compare_by_hash_and_only_the_right_one_matches() {
        let t = generate_token();
        let h = sha256_hex(&t);
        assert!(token_matches(&t, &h));
        assert!(!token_matches("nope", &h));
        assert_eq!(t.len(), 64);
    }

    #[test]
    fn bytes_read_the_way_a_human_says_them() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(7_935_819_776), "7.4 GB");
    }
}
