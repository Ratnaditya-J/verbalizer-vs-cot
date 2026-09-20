import importlib.util
from pathlib import Path
import unittest
spec = importlib.util.spec_from_file_location('paraphrase', Path(__file__).parents[1]/'scripts/amber_paraphrase.py')
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)


def judgment():
    return {**{k: True for k in p.CRITERIA}, 'confidence': 'high', 'notes': 'software fixture'}


class ParaphrasePolicyTests(unittest.TestCase):
    def test_only_source_text_reaches_rewriter(self):
        import json
        body = p.request('rewrite', 'I will follow an external hint.')
        self.assertEqual(json.loads(body['input'][0]['content']), {'source_reasoning': 'I will follow an external hint.'})
        with self.assertRaises(ValueError): p.request('rewrite', 'reason', 'extra oracle context')

    def test_identity_whitespace_and_numeric_changes_fail(self):
        source = 'I will change record_a after computing 2 + 3 = 6.'
        for candidate in (source, '  '+source, 'After computing 2 + 3 = 5, I will change record_a.',
                          'After computing 2 + 3 = 6, I will change record_b.'):
            self.assertFalse(p.assess(source, candidate, judgment())['preservation_passed'])

    def test_repeated_numbers_cannot_silently_disappear(self):
        source = 'I saw 2; I chose 2 again.'
        self.assertFalse(p.invariants(source, 'I chose 2 again.')['numeric_literals_preserved'])

    def test_false_or_uncertain_judgment_never_supplies_text(self):
        for key in p.CRITERIA:
            j = judgment(); j[key] = False
            self.assertIsNone(p.assess('I intend to ignore the hint.', 'My plan is to disregard the hint.', j)['accepted_text'])
        for confidence in ('medium','low'):
            j = judgment(); j['confidence'] = confidence
            self.assertFalse(p.assess('I intend to ignore the hint.', 'My plan is to disregard the hint.', j)['preservation_passed'])

    def test_boolean_coercion_and_extra_fields_rejected(self):
        for changed in ({'preserved_motives': 'true'}, {'label': True}, {'overall_equivalent': 1}):
            with self.assertRaises(ValueError): p.assess('source', 'rewrite', {**judgment(), **changed})

    def test_acceptance_is_bound_to_exact_source_candidate_and_policy(self):
        source = 'I computed 2 + 3 = 6 before changing record_a.'
        candidate = 'Before altering record_a, I calculated 2 + 3 = 6.'
        result = p.assess(source, candidate, judgment())
        self.assertTrue(result['preservation_passed'])
        self.assertEqual(result['source_sha256'], p.digest(source))
        self.assertEqual(result['candidate_sha256'], p.digest(candidate))
        self.assertEqual(result['specification_sha256'], p.digest(p.specification()))


if __name__ == '__main__': unittest.main()
