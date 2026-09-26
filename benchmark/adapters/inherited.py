"""Use the existing aligned evaluator without passing targets or real IDs."""

from common import inherited, require


class Adapter:
    def __init__(self, args):
        self.module = inherited()
        self.evaluator = self.module.Evaluator(args)
        self.runtime = self.evaluator.runtime

    def predict_batch(self, inputs, batch_id):
        rows = []
        for item in inputs:
            require(set(item) == {"state", "qdef"}, "Adapter accepts state/qdef ONLY")
            qdef = item["qdef"]
            # The native sequence helper uses id only in error messages.
            rows.append(dict(id="input", state=item["state"], qdef=qdef,
                             primitive=qdef["type"],
                             options=self.module.render_options(self.module.to_internal(qdef))))
        records, timing = self.evaluator.predict_batch(rows, batch_id)
        require(len(records) == len(inputs), "Backend returned partial batch")
        for record in records:
            for key in ("id", "primitive", "options"):
                record.pop(key, None)
        return records, timing
