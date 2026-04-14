"""Tests for FindingSynthesizer — grouping, ranking, and synthesis."""

from post_training_toolkit.core.finding import Finding
from post_training_toolkit.core.synthesizer import FindingSynthesizer, SynthesizedReport


def _make_finding(type_: str, severity: str = "medium", msg: str = "", rec: str = ""):
    return Finding(type=type_, severity=severity, message=msg or type_, recommendation=rec or None)


class TestSynthesizer:
    def setup_method(self):
        self.synth = FindingSynthesizer()

    def test_empty_findings(self):
        report = self.synth.synthesize([])
        assert report.overall_status == "Stable"
        assert report.total_findings == 0
        assert report.groups == []

    def test_groups_by_keywords(self):
        findings = [
            _make_finding("kl_instability", "high", "KL is unstable"),
            _make_finding("policy_drift", "medium", "Policy drifting"),
            _make_finding("reward_hacking", "high", "Reward correlated with length"),
        ]
        report = self.synth.synthesize(findings)
        group_titles = {g.title for g in report.groups}
        assert "KL / Policy Drift" in group_titles
        assert "Reward Issues" in group_titles

    def test_ungrouped_findings(self):
        findings = [
            _make_finding("some_unknown_issue", "low", "Something unusual"),
        ]
        report = self.synth.synthesize(findings)
        assert len(report.ungrouped) == 1
        assert report.ungrouped[0].type == "some_unknown_issue"

    def test_severity_ranking(self):
        findings = [
            _make_finding("low_issue", "low", "Minor"),
            _make_finding("high_loss_issue", "high", "Loss diverging"),
            _make_finding("medium_entropy", "medium", "Entropy dropping"),
        ]
        report = self.synth.synthesize(findings)
        # Groups should be sorted by severity (high first)
        if report.groups:
            severities = [g.combined_severity for g in report.groups]
            severity_ranks = [{"critical": 0, "high": 1, "medium": 2, "low": 3}.get(s, 5) for s in severities]
            assert severity_ranks == sorted(severity_ranks)

    def test_overall_status_unstable(self):
        findings = [_make_finding("kl_high", "high", "KL too high")]
        report = self.synth.synthesize(findings)
        assert report.overall_status == "Unstable"

    def test_overall_status_partial(self):
        findings = [_make_finding("entropy_warn", "medium", "Entropy declining")]
        report = self.synth.synthesize(findings)
        assert report.overall_status == "Partially unstable"

    def test_overall_status_stable(self):
        findings = [_make_finding("minor_thing", "low", "Small issue")]
        report = self.synth.synthesize(findings)
        assert report.overall_status == "Stable"

    def test_counts(self):
        findings = [
            _make_finding("a", "high"),
            _make_finding("b", "high"),
            _make_finding("c", "medium"),
            _make_finding("d", "low"),
        ]
        report = self.synth.synthesize(findings)
        assert report.total_findings == 4
        assert report.high_count == 2
        assert report.medium_count == 1
        assert report.low_count == 1

    def test_combined_recommendation(self):
        findings = [
            _make_finding("kl_a", "high", "KL issue A", rec="Reduce learning rate."),
            _make_finding("kl_b", "medium", "KL issue B", rec="Increase KL penalty."),
        ]
        report = self.synth.synthesize(findings)
        kl_group = [g for g in report.groups if "KL" in g.title]
        assert len(kl_group) == 1
        assert "Reduce learning rate" in kl_group[0].combined_recommendation
        assert "Increase KL penalty" in kl_group[0].combined_recommendation

    def test_root_cause_hypothesis(self):
        findings = [
            _make_finding("reward_hack_a", "high", "Reward-length correlation"),
            _make_finding("reward_hack_b", "medium", "Reward increasing suspiciously"),
        ]
        report = self.synth.synthesize(findings)
        reward_group = [g for g in report.groups if "Reward" in g.title]
        assert len(reward_group) == 1
        assert reward_group[0].root_cause_hypothesis is not None
