from scripts.analyze_token_length_confounding import clustered_slope, lexical_tokens


def test_lexical_tokens_are_provider_neutral_and_include_punctuation() -> None:
    assert lexical_tokens("Don't stop—now.") == 5


def test_clustered_slope_recovers_within_group_relationship() -> None:
    rows = []
    for question, offset in (("q1", 10.0), ("q2", -4.0), ("q3", 2.0)):
        for model, model_offset in (("a", 3.0), ("b", -1.0)):
            for value in (1.0, 2.0, 3.0):
                rows.append(
                    {
                        "question": question,
                        "model": model,
                        "log2_tokens": value,
                        "outcome": offset + model_offset + 2.0 * value,
                    }
                )
    result = clustered_slope(rows, "outcome", ("question", "model"))
    assert abs(result["slope_per_length_doubling"] - 2.0) < 1e-12
