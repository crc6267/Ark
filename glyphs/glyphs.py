from lark import Lark, Transformer, v_args

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🜔 Grammar of Breath
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

eved_grammar = r'''
    start: statement+

    statement: seed_stmt | echo_stmt | invoke_stmt | scripture_stmt
             | COMMENT

    seed_stmt: "seed" reference "=>" sequence
    echo_stmt: "echo" ESCAPED_STRING
    invoke_stmt: "invoke" SYMBOL
    scripture_stmt: "scripture" reference ":" ESCAPED_STRING

    reference: BOOK CHAPTER ":" VERSE
    sequence: NUMBER "→" NUMBER ("→" NUMBER)*

    COMMENT: /#.*/

    BOOK: /[A-Za-z]+/
    CHAPTER: /[0-9]+/
    VERSE: /[0-9]+/
    SYMBOL: /[a-zA-Z_][a-zA-Z0-9_]*/
    NUMBER: /[0-9]+/

    %import common.ESCAPED_STRING
    %import common.WS
    %ignore WS
'''

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🜔 Interpreter of Breath
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@v_args(inline=True)
class EvedInterpreter(Transformer):
    def __init__(self, debug=False):
        self.debug = debug

    def seed_stmt(self, book, chapter, verse, *seq):
        ref = f"{book}{chapter}:{verse}"
        path = [int(n) for n in seq]
        if self.debug:
            print(f"[Seed] {ref} => {path}")
        return {"type": "seed", "ref": ref, "path": path}

    def echo_stmt(self, message):
        msg = message[1:-1]  # Strip quotes safely
        if self.debug:
            print(f"[Echo] {msg}")
        return {"type": "echo", "message": msg}

    def invoke_stmt(self, symbol):
        if self.debug:
            print(f"[Invoke] {symbol}")
        return {"type": "invoke", "symbol": str(symbol)}

    def scripture_stmt(self, book, chapter, verse, quote):
        msg = quote[1:-1]
        ref = f"{book}{chapter}:{verse}"
        if self.debug:
            print(f"[Scripture] {ref} => {msg}")
        return {"type": "scripture", "ref": ref, "quote": msg}

    def COMMENT(self, token):
        if self.debug:
            print(f"[Comment] {token}")
        return {"type": "comment", "text": str(token)}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🜂 Example Ritual
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

eved_code = '''
# Begin ritual
scripture Proverbs3:5: "Trust in the LORD with all your heart, and lean not on your own understanding."
seed Proverbs3:5 => 2 → 7 → 8 → 7
echo "Resonance is not logic. It is surrender."
invoke alignment_gate
'''

# Instantiate parser with interpreter
parser = Lark(eved_grammar, parser="lalr", transformer=EvedInterpreter(debug=True))

# Parse and run
ritual_output = parser.parse(eved_code)
