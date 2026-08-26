//! Transport-fault gate — a masked hold is a failed cell, never a return series.
//!
//! `sharpebench-sim`'s external transports cannot signal an error through the [`Agent`]
//! trait, so a wire fault returns an empty-orders hold and the fault itself is recorded
//! into a [`TransportHealth`]. Against the native engine an empty-orders decision is a
//! *true* hold: the position persists. A wedged agent therefore rides its last position
//! for the rest of the window, the run completes, and it yields a return series that is
//! indistinguishable from a deliberately conservative agent's.
//!
//! The health record is the only mitigation, and it only mitigates if someone reads it.
//! This module is the reader: it runs an external agent and converts any recorded fault
//! into a typed [`CellOutcome::Failed`]. A failure is evidence; it is never a hold.

use sharpebench_core::Run;
use sharpebench_sim::transport::{DecideError, TransportDiagnostics, TransportHealth};
use sharpebench_sim::{run_backtest, Agent, CostModel, Dataset, Window};

/// Why a run's returns are not scoreable: at least one decision failed at the wire and
/// was masked as an empty-orders hold.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransportFault {
    /// Retryable transport / timeout faults that degraded to a masked hold.
    pub transport_faults: u32,
    /// Agent protocol faults (unparseable output) that degraded to a masked hold.
    pub protocol_faults: u32,
    /// Whether the per-endpoint circuit breaker tripped during the run.
    pub tripped: bool,
    /// The most recent decision-level fault.
    pub last_error: Option<DecideError>,
}

/// The outcome of one external-agent cell: a scoreable run, or a typed failure.
#[derive(Clone, Debug)]
pub enum CellOutcome {
    /// Every decision reached the agent and came back clean. The returns are evidence.
    Scored(Run),
    /// At least one decision was a masked fault. The returns are not evidence.
    Failed(TransportFault),
}

impl CellOutcome {
    /// The run, if it is scoreable. `None` for a failed cell — deliberately not a
    /// default-valued `Run`, so a caller cannot score a failure by accident.
    pub fn scored(&self) -> Option<&Run> {
        match self {
            CellOutcome::Scored(run) => Some(run),
            CellOutcome::Failed(_) => None,
        }
    }

    /// The typed failure, if the cell failed.
    pub fn failure(&self) -> Option<&TransportFault> {
        match self {
            CellOutcome::Scored(_) => None,
            CellOutcome::Failed(fault) => Some(fault),
        }
    }
}

/// Read a rolling health record as a typed failure. `None` means every decision in the
/// run was the agent's own, so the returns may be scored.
pub fn transport_fault(health: &TransportHealth) -> Option<TransportFault> {
    if !health.degraded() {
        return None;
    }
    Some(TransportFault {
        transport_faults: health.transport_faults,
        protocol_faults: health.protocol_faults,
        tripped: health.tripped,
        last_error: health.last_error,
    })
}

/// Run one backtest cell against an external agent and refuse to score it if the
/// transport recorded a fault.
///
/// Use this in place of [`run_backtest`] for anything that speaks over a wire
/// ([`ExternalAgent`], [`HttpAgent`], or any other [`TransportDiagnostics`]
/// implementor). The plain engine call cannot tell a masked hold from a deliberate
/// one; this can.
///
/// [`ExternalAgent`]: crate::ExternalAgent
/// [`HttpAgent`]: crate::HttpAgent
pub fn run_backtest_checked<A>(
    data: &Dataset,
    agent: &mut A,
    window: Window,
    seed: u64,
    costs: CostModel,
) -> CellOutcome
where
    A: Agent + TransportDiagnostics,
{
    let run = run_backtest(data, agent, window, seed, costs);
    match transport_fault(agent.health()) {
        Some(fault) => CellOutcome::Failed(fault),
        None => CellOutcome::Scored(run),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sharpebench_protocol::{Action, Decision, MarketObservation, Order};

    /// An agent whose wire is wedged: it emits the masked empty-orders hold the real
    /// transports emit, and records the fault into its health exactly as they do.
    struct WedgedAgent {
        health: TransportHealth,
        clean_decisions: u32,
    }

    impl WedgedAgent {
        fn wedged_after(clean_decisions: u32) -> Self {
            Self {
                health: TransportHealth::default(),
                clean_decisions,
            }
        }
    }

    impl Agent for WedgedAgent {
        fn decide(&mut self, obs: &MarketObservation) -> Decision {
            if self.clean_decisions > 0 {
                self.clean_decisions -= 1;
                let orders = obs
                    .symbols
                    .iter()
                    .map(|s| Order {
                        symbol: s.symbol.clone(),
                        action: Action::Buy,
                        target_weight: 0.5,
                        confidence: 0.5,
                        rationale: String::new(),
                    })
                    .collect();
                return Decision {
                    orders,
                    reasoning: String::new(),
                    cost: None,
                };
            }
            self.health.record(DecideError::Transport, false);
            Decision {
                orders: Vec::new(),
                reasoning: "external agent transport fault → hold".to_string(),
                cost: None,
            }
        }
    }

    impl TransportDiagnostics for WedgedAgent {
        fn health(&self) -> &TransportHealth {
            &self.health
        }
    }

    /// An agent that never faults, so its health stays clean and its run is scoreable.
    struct HealthyAgent {
        health: TransportHealth,
    }

    impl Agent for HealthyAgent {
        fn decide(&mut self, _obs: &MarketObservation) -> Decision {
            Decision {
                orders: Vec::new(),
                reasoning: "deliberate hold".to_string(),
                cost: None,
            }
        }
    }

    impl TransportDiagnostics for HealthyAgent {
        fn health(&self) -> &TransportHealth {
            &self.health
        }
    }

    #[test]
    fn a_wedged_agent_yields_a_scoreable_series_through_the_unguarded_engine() {
        // The defect this module exists to close: the plain engine call completes and
        // hands back a full return series even though most decisions were masked faults.
        let data = Dataset::synthetic(2, 80, 7);
        let mut agent = WedgedAgent::wedged_after(3);
        let run = run_backtest(
            &data,
            &mut agent,
            Window { start: 10, end: 80 },
            1,
            CostModel::default(),
        );
        assert_eq!(run.returns.len(), 70, "the unguarded run scores in full");
        assert!(
            agent.health().degraded(),
            "and it does so with a degraded transport"
        );
    }

    #[test]
    fn a_wedged_agent_is_a_failed_cell_not_a_return_series() {
        let data = Dataset::synthetic(2, 80, 7);
        let mut agent = WedgedAgent::wedged_after(3);
        let outcome = run_backtest_checked(
            &data,
            &mut agent,
            Window { start: 10, end: 80 },
            1,
            CostModel::default(),
        );
        assert!(
            outcome.scored().is_none(),
            "a masked-fault run must not expose a scoreable series"
        );
        let fault = outcome.failure().expect("the cell is a typed failure");
        assert_eq!(fault.transport_faults, 67);
        assert_eq!(fault.protocol_faults, 0);
        assert_eq!(fault.last_error, Some(DecideError::Transport));
    }

    #[test]
    fn a_clean_transport_is_scored() {
        let data = Dataset::synthetic(2, 80, 7);
        let mut agent = HealthyAgent {
            health: TransportHealth::default(),
        };
        let outcome = run_backtest_checked(
            &data,
            &mut agent,
            Window { start: 10, end: 80 },
            1,
            CostModel::default(),
        );
        assert!(outcome.failure().is_none(), "no fault was recorded");
        assert_eq!(
            outcome.scored().expect("scoreable").returns.len(),
            70,
            "a deliberate hold is still evidence"
        );
    }

    #[test]
    fn a_protocol_fault_is_also_a_failed_cell() {
        let mut health = TransportHealth::default();
        health.record(DecideError::Protocol, false);
        let fault = transport_fault(&health).expect("a protocol fault is a fault");
        assert_eq!(fault.protocol_faults, 1);
        assert_eq!(fault.transport_faults, 0);
        assert!(!fault.tripped);
    }

    #[test]
    fn an_undegraded_health_reads_as_no_fault() {
        assert_eq!(transport_fault(&TransportHealth::default()), None);
    }
}
