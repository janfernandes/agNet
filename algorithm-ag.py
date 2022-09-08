import igraph as ig
import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.model_selection import train_test_split


def pagerank(graph):
    return ig.Graph.pagerank(graph)


def create_representation(X_train, y_train, q_value):
    euclidean_dist = euclidean_distances(X_train)
    np.fill_diagonal(euclidean_dist, np.inf)
    neighbors = np.argsort(euclidean_dist, axis=1)[:, :q_value]
    mask = np.zeros((len(X_train), q_value)).astype(int)
    for i in range(len(neighbors)):
        mask[i] = (y_train[neighbors[i]] == y_train[i])
    return neighbors, mask


def generate_ind(genome_size, q_value):
    if q_value == 5:
        return [[np.random.randint(2), np.random.randint(2), np.random.randint(2), np.random.randint(2), np.random.randint(2)] for _ in range(genome_size)]
    return [[np.random.randint(2), np.random.randint(2), np.random.randint(2)] for _ in range(genome_size)]


def create_population(genome_size, q_value, TP, CR):
    Pop = np.zeros((TP + CR, genome_size, q_value)).astype(int)
    Fit = np.zeros(TP + CR)

    Pop[:TP] = [generate_ind(genome_size, q_value) for _ in range(TP)]

    return Pop, Fit


def validate_pop(pop, actual_classes):
    # os que nao percecem a claase nao recebem a conecao
    not_connect = np.argwhere(actual_classes == 0)
    for value in not_connect:
        pop[:, value[0]][:, value[1]] = 0


def generate_graph(X_train, individuo, neighbors, q_value):
    euclidean_dist = euclidean_distances(X_train)
    sources = np.arange(0, len(individuo))

    graph = ig.Graph(n=len(sources), directed=True)
    for i in sources:
        for ids_dest in range(q_value):
            # decide estocasticamente se deve haver a conexao
            if individuo[i][ids_dest] == 1:
                graph.add_edge(i, neighbors[i][ids_dest], weight=euclidean_dist[i, neighbors[i][ids_dest]])
    return graph


# requer a passagem de um grafo ponderado como parametro
def efficiency_flow(graph):
    global edge_sources
    eff = np.zeros(graph.vcount())
    try:
        edge_sources = np.asarray(graph.get_edgelist()).T[0]
    except:
        print("An exception occurred")
    for j, i in enumerate(edge_sources):
        eff[i] += graph.es['weight'][j]

    count = np.bincount(edge_sources)

    for i in range(len(count)):
        if count[i] != 0:
            eff[i] = eff[i] / count[i]

    comps = ig.Graph.components(graph, mode=WEAK)
    for i in range(len(comps)):
        comp_ids = np.array(comps[i])
        if len(comp_ids) > 1:
            eff[comp_ids] = np.sum(eff[comp_ids]) / (1. * len(comp_ids))

    return eff


def evaluate_config(X_train, y_train, X_val, y_val, pop, neighbor, actual_classes, alpha, q_value):
    fit = []
    graphs = []
    # os que nao forem da msm classe nao conecta
    validate_pop(pop, actual_classes)
    for individual in pop:
        graph = generate_graph(X_train, individual, neighbor, q_value)
        graphs.append(graph)
        # se gerou um grafo totalmente desconexo
        if len(graph.get_edgelist()) == 0:
            fit = np.append(fit, 0)
        else:
            eff = efficiency_flow(graph)
            fdist = euclidean_distances(X_val, X_train)

            I = np.asarray(pagerank(graph))

            for z1, a in enumerate(alpha):
                prob_class = np.zeros((len(X_val), len(np.unique(y_val))))
                for i in range(len(X_val)):
                    f = eff[i] * a - fdist[i, :]
                    ids = np.where(f >= 0.)[0]

                    if len(ids) == 0:
                        ids = np.where(f == max(f))[0]

                    for j in ids:
                        prob_class[i, y_train[j]] += I[j]

                predicted = np.argmax(prob_class, axis=1)
                ac = np.mean(predicted == y_val)
                fit = np.append(fit, ac)
    return fit, graphs


def tournament(Fit, CR, TP, TOURNAMENT_SIZE):
    parents = [do_tournament(Fit, TP, TOURNAMENT_SIZE) for _ in range(0, CR, 2)]
    return np.array(parents).reshape(1, -1)[0]


def roulette(Fit, TP, CR):
    fit_invertido = (max(Fit[:TP]) + 1) - Fit[:TP]
    max_v = sum(fit_invertido).astype(np.float64)
    parents = np.random.choice(TP, CR, p=(None if max_v == 0 else fit_invertido / max_v))
    return parents


def do_tournament(Fit, TP, TOURNAMENT_SIZE):
    random_parents = np.random.randint(TP, size=TOURNAMENT_SIZE)
    parent1 = random_parents[np.argmax([Fit[random_parents]])]
    random_parents = np.random.randint(TP, size=TOURNAMENT_SIZE)
    parent2 = random_parents[np.argmax([Fit[random_parents]])]
    return parent1, parent2


def apply_mutation(pop, genome_size, q_value, TP, CR, PMUT):
    ind_to_mutate = np.random.randint(low=TP, high=TP + CR, size=int(TP * PMUT))
    for individual in ind_to_mutate:
        genes = np.random.choice(genome_size, size=int(genome_size * PMUT))
        for gene in genes:
            # mutando co certeza
            for qi in range(q_value):
                pop[individual][gene][qi] = 1 if genes[0] == 0 else 0


def ordered_reinsertion(Pop, Fit, TP):
    aux_pop = np.zeros(Pop.shape).astype(int)
    fit_sorted = np.argsort(-Fit)[:TP]
    aux_pop[:TP] = Pop[fit_sorted]
    return aux_pop


def pure_reinsertion(Pop, Fit, TP, CR):
    aux_pop = np.zeros(Pop.shape).astype(int)
    fit_sorted = np.argsort(Fit[:TP])[:TP - CR]
    aux_pop[:TP - CR] = np.copy(Pop[fit_sorted])
    aux_pop[TP - CR:TP] = np.copy(Pop[TP:TP + CR])
    return aux_pop


def two_points_crossover(parent1, parent2, genome_size):
    gene1, gene2 = np.random.choice(genome_size, 2, replace=False)
    new_child1 = np.copy(parent1)
    new_child2 = np.copy(parent2)
    aux = np.copy(parent1)
    if gene1 > gene2:
        gene1, gene2 = gene2, gene1

    new_child1[range(gene1, gene2)] = np.copy(new_child2[range(gene1, gene2)])
    new_child2[range(gene1, gene2)] = np.copy(aux[range(gene1, gene2)])
    return new_child1, new_child2


def mask_crossover(parent1, parent2, genome_size):
    mask = np.random.randint(low=0, high=2, size=genome_size)
    new_child1 = np.copy(parent1)
    new_child2 = np.copy(parent2)

    new_child1[np.where(mask == 1)] = parent2[np.where(mask == 1)]
    new_child2[np.where(mask == 0)] = parent1[np.where(mask == 0)]

    return new_child1, new_child2


def one_point_crossover(parent1, parent2, genome_size):
    gene = np.random.choice(genome_size, replace=False)
    new_child1 = np.copy(parent1)
    new_child2 = np.copy(parent2)
    aux = np.copy(parent1)

    new_child1[range(0, gene)] = np.copy(new_child2[range(0, gene)])
    new_child2[range(gene, genome_size)] = np.copy(aux[range(gene, genome_size)])
    return new_child1, new_child2


def alg(config, X, y, random, alpha, q_value, EXECUTIONS, TP, CR, GEN, PMUT):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=np.random.seed(random * 100))
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=np.random.seed(random * 100))
    neighbors, actual_classes = create_representation(X_train, y_train, q_value)

    genome_size = X_train.shape[0]

    for execution in range(EXECUTIONS):
        pop, fit = create_population(genome_size, q_value, TP, CR)
        # testando a opcao de fazer o cut somente qndo avalia a populacao
        fit[:TP], graphs = evaluate_config(X_train, y_train, X_val, y_val, pop[:TP], neighbors, actual_classes, alpha, q_value)

        for generation in range(GEN):
            parents = config[3](fit[:TP])

            for i in range(0, CR - 1, 2):
                # pensar como fazer o crossover d forma simples
                pop[TP + i], pop[TP + i + 1] = config[5](pop[parents[i]], pop[parents[i + 1]], genome_size)

            # pensar na mutacao
            apply_mutation(pop, genome_size, q_value, TP, CR, PMUT)

            # avaliacao
            fit, graphs = evaluate_config(X_train, y_train, X_val, y_val, pop, neighbors, actual_classes, alpha, q_value)

            # Re-insercao
            pop = np.copy(config[4](pop, fit))
            fit, graphs = evaluate_config(X_train, y_train, X_val, y_val, pop[:TP], neighbors, actual_classes, alpha, q_value)

        acc_val = fit[np.argmax(fit[:TP])]

        acc_teste, graph_final = evaluate_config(X_train, y_train, X_test, y_test,
                                                 pop[np.argmax(fit[:TP])].reshape(1, len(pop[np.argmax(fit[:TP])]), -1), neighbors,
                                                 actual_classes, alpha, q_value)

        return acc_val, acc_teste[0], graph_final[0]
