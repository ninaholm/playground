from app import db


class TWSSsentence(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sentence = db.Column(db.String(1000))
    timestamp = db.Column(db.DateTime)
    isTwss = db.Column(db.Boolean)

    def __repr__(self):
        return self.sentence

