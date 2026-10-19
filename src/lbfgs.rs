//! Limited-memory BFGS, for fitting the model in `norn learn` without a dependency.

const HISTORY: usize = 10;

/// Minimises `f` from `x`. `f` returns the value at a point and writes the gradient into
/// its second argument. Stops after `max_iter` iterations, or once the gradient or the
/// improvement per step becomes negligible.
pub fn minimize<F>(mut x: Vec<f64>, mut f: F, max_iter: usize) -> Vec<f64>
where
    F: FnMut(&[f64], &mut [f64]) -> f64,
{
    let n = x.len();
    let mut g = vec![0.0; n];
    let mut value = f(&x, &mut g);
    let mut s_hist: Vec<Vec<f64>> = Vec::new();
    let mut y_hist: Vec<Vec<f64>> = Vec::new();
    let mut x_new = vec![0.0; n];
    let mut g_new = vec![0.0; n];

    for iter in 0..max_iter {
        if norm_inf(&g) < 1e-6 {
            break;
        }
        let d = direction(&g, &s_hist, &y_hist);
        let slope = dot(&g, &d);
        if slope >= 0.0 {
            // Not a descent direction: start again from steepest descent.
            s_hist.clear();
            y_hist.clear();
            continue;
        }
        // The first step has no curvature information, so keep it short.
        let mut step = if iter == 0 {
            1.0 / norm2(&g).max(1.0)
        } else {
            1.0
        };
        let new_value = loop {
            for i in 0..n {
                x_new[i] = x[i] + step * d[i];
            }
            let v = f(&x_new, &mut g_new);
            if v <= value + 1e-4 * step * slope || step < 1e-12 {
                break v;
            }
            step *= 0.5;
        };

        let s: Vec<f64> = (0..n).map(|i| x_new[i] - x[i]).collect();
        let y: Vec<f64> = (0..n).map(|i| g_new[i] - g[i]).collect();
        if dot(&s, &y) > 1e-12 {
            if s_hist.len() == HISTORY {
                s_hist.remove(0);
                y_hist.remove(0);
            }
            s_hist.push(s);
            y_hist.push(y);
        }
        let improvement = value - new_value;
        std::mem::swap(&mut x, &mut x_new);
        std::mem::swap(&mut g, &mut g_new);
        value = new_value;
        if improvement.abs() <= 1e-10 * value.abs().max(1.0) {
            break;
        }
    }
    x
}

/// The two-loop recursion: an approximation of -H⁻¹g from recent steps.
fn direction(g: &[f64], s_hist: &[Vec<f64>], y_hist: &[Vec<f64>]) -> Vec<f64> {
    let mut q: Vec<f64> = g.iter().map(|v| -v).collect();
    let rho: Vec<f64> = s_hist
        .iter()
        .zip(y_hist)
        .map(|(s, y)| 1.0 / dot(s, y))
        .collect();
    let mut alpha = vec![0.0; s_hist.len()];
    for i in (0..s_hist.len()).rev() {
        alpha[i] = rho[i] * dot(&s_hist[i], &q);
        axpy(-alpha[i], &y_hist[i], &mut q);
    }
    if let (Some(s), Some(y)) = (s_hist.last(), y_hist.last()) {
        let gamma = dot(s, y) / dot(y, y);
        q.iter_mut().for_each(|v| *v *= gamma);
    }
    for i in 0..s_hist.len() {
        let beta = rho[i] * dot(&y_hist[i], &q);
        axpy(alpha[i] - beta, &s_hist[i], &mut q);
    }
    q
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn axpy(a: f64, x: &[f64], y: &mut [f64]) {
    y.iter_mut().zip(x).for_each(|(y, x)| *y += a * x);
}

fn norm2(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

fn norm_inf(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, x| m.max(x.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimises_a_quadratic() {
        // f = sum (i+1) (x_i - i)^2
        let x = minimize(
            vec![10.0; 5],
            |x, g| {
                let mut v = 0.0;
                for i in 0..x.len() {
                    let d = x[i] - i as f64;
                    v += (i + 1) as f64 * d * d;
                    g[i] = 2.0 * (i + 1) as f64 * d;
                }
                v
            },
            100,
        );
        for (i, xi) in x.iter().enumerate() {
            assert!((xi - i as f64).abs() < 1e-5, "{x:?}");
        }
    }

    #[test]
    fn minimises_rosenbrock() {
        let x = minimize(
            vec![-1.2, 1.0],
            |x, g| {
                let (a, b) = (x[0], x[1]);
                g[0] = -2.0 * (1.0 - a) - 400.0 * a * (b - a * a);
                g[1] = 200.0 * (b - a * a);
                (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2)
            },
            500,
        );
        assert!(
            (x[0] - 1.0).abs() < 1e-3 && (x[1] - 1.0).abs() < 1e-3,
            "{x:?}"
        );
    }
}
