import re
import random
from typing import Dict, List, Tuple

# Algorithme CKY
class CKY:
    def __init__(self, gram: List[Tuple[str, str, str]], lex: Dict[str, List[str]]):
        self.gram = gram 
        self.lex = lex 

    def parse(self, sent: List[str]) -> List[List[List[Tuple[str, int, int, int]]]]:
        N = len(sent)
        T = [[[] for _ in range(N)] for _ in range(N)]
        
        # Remplissage de la diagonale avec les règles lexicales
        for i in range(N):
            word = sent[i]
            for A in self.lex.get(word, []):
                T[i][i].append((A, -1, -1, -1))
            
            # Application des règles unitaires
            added = True
            while added:
                added = False
                current_entries = T[i][i].copy()
                for entry_idx, entry in enumerate(current_entries):
                    B = entry[0]
                    for rule in self.gram:
                        if len(rule) == 2 and rule[1] == B:
                            A = rule[0]
                            # Vérifier si cette règle a déjà été appliquée
                            if not any(existing[0] == A and existing[2] == entry_idx for existing in T[i][i]):
                                T[i][i].append((A, i, entry_idx, -1))
                                added = True
        
        # Algorithme CKY
        for span in range(2, N + 1):
            for i in range(N - span + 1):
                j = i + span - 1
                
                # Appliquer toutes les règles binaires
                for k in range(i, j):
                    for B_idx, B_entry in enumerate(T[i][k]):
                        B = B_entry[0]
                        for C_idx, C_entry in enumerate(T[k+1][j]):
                            C = C_entry[0]
                            
                            # Chercher toutes les règles A -> B C
                            for rule in self.gram:
                                if len(rule) == 3 and rule[1] == B and rule[2] == C:
                                    A = rule[0]
                                    # Vérifier s'il n'existe pas déjà un résultat similaire
                                    if not any(existing[0] == A and existing[1] == k and 
                                            existing[2] == B_idx and existing[3] == C_idx 
                                            for existing in T[i][j]):
                                        T[i][j].append((A, k, B_idx, C_idx))
                
                # Appliquer les règles unitaires
                added = True
                while added:
                    added = False
                    current_entries = T[i][j].copy()
                    for entry_idx, entry in enumerate(current_entries):
                        B = entry[0]
                        for rule in self.gram:
                            if len(rule) == 2 and rule[1] == B:
                                A = rule[0]
                                # Vérifier si cette règle n'a pas déjà été appliquée
                                if not any(existing[0] == A and existing[2] == entry_idx for existing in T[i][j]):
                                    T[i][j].append((A, j, entry_idx, -1))
                                    added = True
        
        return T

    def export_json(self):
        return self.__dict__.copy()

    def import_json(self, data):
        for key in data:
            self.__dict__[key] = data[key]

# Fonction d'évaluation
def pr_eval(ref, sys):
    if ref is None and sys is None:
        return 1.0, 1.0
    if ref is None or sys is None:
        return 0.0, 0.0

    def extract_arcs(tree, index=0):
        if not isinstance(tree, tuple):
            return [], index

        arcs = []
        current = (tree[0], index)  
        next_index = index + 1

        for child in tree[1:]:
            if isinstance(child, tuple):
                child_label = (child[0], next_index)  
                arcs.append((current, child_label))
                child_arcs, next_index = extract_arcs(child, next_index)
                arcs.extend(child_arcs)
            else:
                next_index += 1

        return arcs, next_index

    ref_arcs, _ = extract_arcs(ref)
    sys_arcs, _ = extract_arcs(sys)

    common_arcs = set(ref_arcs) & set(sys_arcs)
    true_positives = len(common_arcs)

    precision = true_positives / len(sys_arcs) if sys_arcs else 0.0
    recall = true_positives / len(ref_arcs) if ref_arcs else 0.0

    return precision, recall

def construct(T, sent, i, j, pos):
    A, k, iB, iC = T[i][j][pos]
    if k >= 0:
        left = construct(T, sent, i, k, iB)
        if iC == -1:
            return (A, left)
        right = construct(T, sent, k+1, j, iC)
        return (A, left, right)
    return (A, sent[i])

def parse_tuple(string):
    string = re.sub(r'([^\s(),]+)', "'\\1'", string)
    try:
        s = eval(string)
        if type(s) == tuple:
            return s
        return
    except:
        return

class Syntax():
    def __init__(self):
        self.eval = []

    def _parse(self, sent):
        r = None
        T = self.model.parse(sent)
        n = len(sent) - 1
        for pos in range(len(T[0][n])):
            if T[0][n][pos][0] == 'S':
                r = construct(T, sent, 0, n, pos)
                break
        return r

    def parse(self, sent: str):
        return self._parse(sent.strip().lower().split())

    def load_model(self, url):
        f = open(url, 'r')
        lex = {}
        gram = []
        for l in f:
            l = l.strip()
            if len(l) < 3 or l.startswith('#') :
                continue
            info = l.split('	')
            if len(info) == 2:
                if info[1][0].isupper():
                    gram.append((info[0], info[1]))
                else:
                    if not info[1] in lex:
                        lex[info[1]] = []
                    lex[info[1]].append(info[0])
            elif len(info) == 3:
                gram.append((info[0], info[1], info[2]))
        self.model = CKY(gram, lex)
        f.close()

    def load_eval(self, url):
        f = open(url, 'r')
        for l in f: 
            l = l.strip()
            if len(l) < 5 or l.startswith('#'):
                continue
            info = l.split('	')
            
            self.eval.append((info[0], parse_tuple(info[1])))

    def evaluate(self, n):
        if n == -1:
            S = self.eval
            n = len(S)
        else :
            S = random.sample(self.eval, n)
        P, R = 0.0, 0.0
        for i in range(n):
            test = S[i]
            print('=======================')
            print('sent:', test[0])
            print('ref tree:', test[1])
            tree = self.parse(test[0])
            print('sys tree:', tree)
            P_i, R_i = pr_eval(test[1], tree)
            print('P, R:', P_i, R_i)
            P += P_i
            R += R_i

        P, R = P/n, R/n
        print('---------------------------------')
        print('P, R: ', P, R)

# Génération de l'arbre
def generate_node(node, id=0):
    if node is None:
        return 0, ''
    nid = id + 1
    if len(node) < 3:
        return nid, 'N' + str(id) + '[label="' + node[0] + "=" + node[1] + '" shape=box];\n'
    res = 'N' + str(id) + '[label="' + node[0] + '"];\n'
    nid_l = nid
    nid, code = generate_node(node[1], id=nid_l)
    res += code
    res += 'N' + str(id) + ' ->  N' + str(nid_l) + ';\n'
    if len(node) > 2:
        nid_r = nid
        nid, code = generate_node(node[2], id=nid_r)
        res += code
        res += 'N' + str(id) + ' ->  N' + str(nid_r) + ';\n'
    return nid, res

def generate_graphviz(root, url):
    res = 'digraph Tree {\n'
    id, code = generate_node(root)
    res += code
    res += '}'
    f = open(url, 'w')
    f.write(res)
    f.close()

# Test de l'algorithme CKY
def test_cky():
    parser = Syntax()
    parser.load_model('data/grammaire.txt')
    sent = 'la petite forme une petite phrase'
    result = "('S', ('NP', ('D', 'la'), ('N', 'petite')), ('VP', ('V', 'forme'), ('NP', ('D', 'une'), ('AP', ('J', 'petite'), ('N', 'phrase')))))"
    tree = parser.parse(sent)
    print('Real Result: ', result)
    print('My Result: ', tree)
    generate_graphviz(tree, 'arbre.gv')

# Test de l'évaluation de l'arbre
def test_eval_tree():
    t1 = ('A', ('B', 'b'), ('C', ('A', 'a'), ('B', ('A', 'a'), ('C', 'c'))))
    t2 = ('A', ('B', 'b'), ('C', ('B', 'b'), ('D', 'd')))
    t3 = ('A', ('B', 'b'), ('C', ('B', 'b')))
    print('Real: (0., 0.), Found: ', pr_eval(None, t1))
    print('Real: (1., 1.), Found: ', pr_eval(None, None))
    print('Real: (0.5, 0.333), Found: ', pr_eval(t1, t2))
    print('Real: (0.333, 0.5), Found: ', pr_eval(t2, t1))
    print('Real: (1., 0.75), Found: ', pr_eval(t2, t3))
    print('Real: (1., 1.), Found: ', pr_eval(t1, t1))

# Test final
def test_evaluate():
    parser = Syntax()
    parser.load_model('data/grammaire.txt')
    parser.load_eval('data/data_test.txt')
    parser.evaluate(-1)

if __name__ == '__main__':
    test_cky()
    test_eval_tree()
    test_evaluate()