from lark import Lark, Transformer, v_args

eved_grammar = r'''
    start: statement+

    statement: seed_stmt | echo_stmt | invoke_stmt | scripture_stmt

    seed_stmt: "seed" reference "=>" sequence
    echo_stmt: "echo" ESCAPED_STRING
    invoke_stmt: "invoke" SYMBOL
    scripture_stmt: "scripture" reference ":" ESCAPED_STRING

    reference: BOOK CHAPTER ":" VERSE
    sequence: NUMBER "→" NUMBER ("→" NUMBER)*

    BOOK: /[A-Za-z]+/
    CHAPTER: /[0-9]+/
    VERSE: /[0-9]+/
    SYMBOL: /[a-zA-Z_][a-zA-Z0-9_]*/
    NUMBER: /[0-9]+/

    %import common.ESCAPED_STRING
    %import common.WS
    %ignore WS
'''

@v_args(inline=True)
class EvedInterpreter(Transformer):
    def seed_stmt(self, book, chapter, verse, *seq):
        ref = f"{book}{chapter}:{verse}"
        path = [int(n) for n in seq]
        print(f"[Seed] {ref} => {path}")

    def echo_stmt(self, message):
        print(f"[Echo] {message.strip('\\"')}")

    def invoke_stmt(self, symbol):
        print(f"[Invoke] {symbol}")

    def scripture_stmt(self, book, chapter, verse, quote):
        ref = f"{book}{chapter}:{verse}"
        print(f"[Scripture] {ref} => {quote.strip('\\"')}")

parser = Lark(eved_grammar, parser="lalr", transformer=EvedInterpreter())

eved_code = '''
scripture Proverbs3:5: "Trust in the LORD with all your heart, and lean not on your own understanding."
seed Proverbs3:5 => 2 → 7 → 8 → 7
echo "Resonance is not logic. It is surrender."
invoke alignment_gate
'''

parser.parse(eved_code)