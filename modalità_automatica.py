import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, HTML
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
from scipy.stats import chi2
import numpy as np

dataset = pd.read_excel("heart.xlsx")
gum.config["notebook", "default_arc_color"] = "#4682B4"
gum.config["notebook", "default_node_bgcolor"] = "#4682B4"
gum.config["notebook", "default_node_fgcolor"] = "#FFFFFF"

variables = dataset.columns.tolist()

dropdown = widgets.Dropdown(
    options=variables,
    description='Variabile:',
)

score_method_dropdown = widgets.Dropdown(
    options=['Seleziona un metodo', 'AIC', 'BIC', 'BDeu'],
    description='Scoring:',
    value='Seleziona un metodo',
)

dropdown_target = widgets.Dropdown(
    options=dataset.columns.tolist(),
    description='Evento:',
)

dropdown_condition = widgets.Dropdown(
    options=dataset.columns.tolist(),
    description='Evidenza:',
)

dropdown_condition_value = widgets.Dropdown(
    description='Valore:',
)
from_input = widgets.Text(description='Da:', placeholder='Nodo di partenza')
to_input = widgets.Text(description='A:', placeholder='Nodo di arrivo')
button_width = '200px'
input_width = '300px'

dropdown_discretize_value = widgets.Dropdown(
    description='Valore:',
)

text_new_value = widgets.Text(
    description='Nuovo Valore:',
    placeholder='Inserisci nuovo valore',
)

button_check_missing = widgets.Button(
    description='Controlla Dati Mancanti',
    style={'button_color': '#87CEEB', 'font_weight': 'bold', 'font_color': '#FFFFFF'},
    layout=widgets.Layout(width=button_width)
)
button_show_distribution = widgets.Button(
    description="Visualizza Grafico",
    style={'button_color': '#87CEEB', 'font_weight': 'bold', 'font_color': '#FFFFFF'},
    layout=widgets.Layout(width=button_width)
)
button_add_blacklist = widgets.Button(
    description="Blacklist",
    style={'button_color': '#87CEEB', 'font_weight': 'bold', 'font_color': '#FFFFFF'},
    layout=widgets.Layout(width=button_width)
)
button_add_whitelist = widgets.Button(
    description="Whitelist",
    style={'button_color': '#87CEEB', 'font_weight': 'bold', 'font_color': '#FFFFFF'},
    layout=widgets.Layout(width=button_width)
)
button_learn = widgets.Button(
    description='Addestra Rete',
    style={'button_color': '#87CEEB', 'font_weight': 'bold', 'font_color': '#FFFFFF'},
    layout=widgets.Layout(width=button_width)
)
button_strength = widgets.Button(
    description='Calcola Strength',
    style={'button_color': '#87CEEB', 'font_weight': 'bold', 'font_color': '#FFFFFF'},
    layout=widgets.Layout(width=button_width)
)
button_show_cpt = widgets.Button(
    description='Mostra CPT',
    style={'button_color': '#87CEEB', 'font_weight': 'bold', 'font_color': '#FFFFFF'},
    layout=widgets.Layout(width=button_width)
)

button_inference = widgets.Button(
    description="Esegui Inferenza",
    style={'button_color': '#87CEEB', 'font_weight': 'bold', 'font_color': '#FFFFFF'},
    layout=widgets.Layout(width=button_width)
)

output = widgets.Output()
output_missing_data = widgets.Output()
output_distribution = widgets.Output()
output_bayes_net = widgets.Output()
output_inference = widgets.Output()
blacklist =[]
whitelist =[]

selected_bayes_net = None

def check_missing_data():
    missing_data = dataset.isnull().sum()
    missing_percentage = (missing_data / len(dataset)) * 100
    missing_info = pd.DataFrame({
        'Valori Mancanti': missing_data,
        'Percentuale': missing_percentage
    })
    missing_info = missing_info[missing_info['Valori Mancanti'] > 0].sort_values('Valori Mancanti', ascending=False)
    return missing_info

def display_missing_data(b):
    with output_missing_data:
        output_missing_data.clear_output()
        missing_info = check_missing_data()
        if missing_info.empty:
            print("✅ Non ci sono dati mancanti nel dataset!")
        else:
            display(HTML("<h3>Dati Mancanti nel Dataset:</h3>"))
            display(missing_info.style.format({
                'Percentuale': '{:.2f}%'
            }))

def show_distribution(data, title="Grafico", bins=30):
    plt.figure(figsize=(10, 6))
    sns.histplot(data, bins=bins)
    plt.title(title)
    plt.xlabel("Valori")
    plt.ylabel("Frequenza")
    plt.show()
    summary = data.describe()
    print(f"Summary di {title}:\n{summary}\n")

def display_distribution(b):
    with output_distribution:
        output_distribution.clear_output()
        show_distribution(dataset[dropdown.value], title=f"Grafico di {dropdown.value}")

def add_to_blacklist(b):
    with output:
        output.clear_output()
        if selected_bayes_net is None:
            print("❌ Errore: prima apprendi una rete bayesiana.")
            return
        
        valid_names = selected_bayes_net.names()
        from_var = from_input.value
        to_var = to_input.value

        if from_var not in valid_names or to_var not in valid_names:
            missing_vars = [var for var in [from_var, to_var] if var not in valid_names]
            print(f"❌ Errore: le seguenti variabili non esistono nella rete bayesiana: {', '.join(missing_vars)}")
            print(f"Variabili disponibili: {', '.join(valid_names)}")
            return

        blacklist.append((from_var, to_var))
        print(f"✅ Aggiunto alla blacklist: {from_var} -> {to_var}")

def add_to_whitelist(b):
    with output:
        output.clear_output()
        if selected_bayes_net is None:
            print("❌ Errore: prima apprendi una rete bayesiana.")
            return
        
        valid_names = selected_bayes_net.names()
        from_var = from_input.value
        to_var = to_input.value

        if from_var not in valid_names or to_var not in valid_names:
            missing_vars = [var for var in [from_var, to_var] if var not in valid_names]
            print(f"❌ Errore: le seguenti variabili non esistono nella rete bayesiana: {', '.join(missing_vars)}")
            print(f"Variabili disponibili: {', '.join(valid_names)}")
            return

        whitelist.append((from_var, to_var))
        print(f"✅ Aggiunto alla whitelist: {from_var} -> {to_var}")

learner = None

def learn_bayes_net(score_method):
    global learner, selected_bayes_net
    
    learner = gum.BNLearner(dataset)
    learner.useGreedyHillClimbing()

    if score_method == 'AIC':
        learner.useScoreAIC()
        learner.useSmoothingPrior(weight=0.1)
    elif score_method == 'BIC':
        learner.useScoreBIC()
        learner.useSmoothingPrior(weight=0.1)
    elif score_method == 'BDeu':
        learner.useScoreBDeu()
        
    for arc in blacklist:
        learner.addForbiddenArc(arc[0], arc[1])
    for arc in whitelist:
        learner.addMandatoryArc(arc[0], arc[1])

    bayes_net = learner.learnBN()

    global_score = 0
    for node in bayes_net.nodes():
        node_name = bayes_net.variable(node).name()
        parents = [bayes_net.variable(p).name() for p in bayes_net.parents(node)]

        valid_parents = [
            parent for parent in parents 
            if (parent, node_name) not in blacklist
        ]
        
        node_score = learner.score(node_name, valid_parents)
        global_score += node_score

    num_arcs = bayes_net.dag().sizeArcs()

    num_nodes = bayes_net.size()
    global_score =  global_score / num_nodes
    avg_markov_size = sum([gum.MarkovBlanket(bayes_net, node).sizeNodes() - 1 for node in bayes_net.nodes()]) / num_nodes
    
    avg_neighbourhood_size = sum([
        len(set(bayes_net.parents(node)).union(set(bayes_net.children(node))))
        for node in bayes_net.nodes()
    ]) / num_nodes
    

    return bayes_net, global_score, num_arcs, avg_markov_size, avg_neighbourhood_size

def learn_and_display_all_scores(b):
    global selected_bayes_net

    with output_bayes_net:
        output_bayes_net.clear_output()

        display(HTML("<h3>Blacklist:</h3>"))
        if blacklist:
            for arc in blacklist:
                display(HTML(f"<div>{arc[0]} -> {arc[1]}</div>"))
        else:
            display(HTML("<div>Nessun arco nella blacklist.</div>"))

        display(HTML("<h3>Whitelist:</h3>"))
        if whitelist:
            for arc in whitelist:
                display(HTML(f"<div>{arc[0]} -> {arc[1]}</div>"))
        else:
            display(HTML("<div>Nessun arco nella whitelist.</div>"))

        bayes_nets = []
        captions = []
        scores = {}

        for method in ['AIC', 'BIC', 'BDeu']:
            bn, score, num_arcs, avg_markov_size, avg_neighbourhood_size = learn_bayes_net(method)
            scores[method] = (score, num_arcs, avg_markov_size, avg_neighbourhood_size)
            bayes_nets.append(bn)
            captions.append(
                f"Hill Climbing con {method} (Score: {score:.2f}, Arcs: {num_arcs}, Avg Markov: {avg_markov_size:.2f})"
            )

        gnb.sideBySide(*bayes_nets, captions=captions)

        display(HTML("<h3>Risultati dei Metodi:</h3>"))
        for method, (score, num_arcs, avg_markov_size, avg_neighbourhood_size) in scores.items():
            display(HTML(f"<div style='background-color: lightblue; padding: 10px;'>"
                         f"<strong>Metodo:</strong> {method}<br>"
                         f"<strong>Score:</strong> {score:.2f}<br>"
                         f"<strong>Numero Archi:</strong> {num_arcs}<br>"
                         f"<strong>Media Markov Blanket:</strong> {avg_markov_size:.2f}<br>"
                         f"<strong>Media Neighbourhood:</strong> {avg_neighbourhood_size:.2f}<br></div>"))

        if score_method_dropdown.value != 'Seleziona un metodo':
            selected_bayes_net, _, _, _, _ = learn_bayes_net(score_method_dropdown.value)
            print(f"Rete selezionata con il metodo '{score_method_dropdown.value}' salvata per analisi successive.")

def calculate_arc_strengths(b):
    with output_bayes_net:
        output_bayes_net.clear_output()

        if not selected_bayes_net:
            print("Errore: prima apprendi una rete bayesiana.")
            return

        arc_strengths = []
        N = len(dataset)
        LOG2 = 0.693147

        for arc in selected_bayes_net.arcs():
            from_node = selected_bayes_net.variable(arc[0]).name()
            to_node = selected_bayes_net.variable(arc[1]).name()
            
            try:
                mi = learner.mutualInformation(from_node, to_node)
                
                df = (len(dataset[from_node].unique()) - 1) * (len(dataset[to_node].unique()) - 1)
                
                mi_stat = 2 * N * mi * LOG2
                
                p_value = chi2.sf(mi_stat/2, df)
                
                arc_strengths.append({
                    "from": from_node,
                    "to": to_node,
                    "p_value": p_value
                })
            except Exception as e:
                print(f"❌ Errore nel calcolo per l'arco {from_node} -> {to_node}: {e}")
                continue

        if arc_strengths:
            df_strengths = pd.DataFrame(arc_strengths)
            df_strengths = df_strengths.sort_values('p_value', ascending=False)
            
            df_strengths['p_value'] = df_strengths['p_value'].apply(
                lambda x: f"{x:.6e}"
            )
            
            display(HTML("<h3>Forza dell'arco:</h3>"))
            display(df_strengths)
        else:
            print("Nessun arco presente nella rete.")

def show_cpt(b):
    with output_bayes_net:
        output_bayes_net.clear_output()

        if not selected_bayes_net:
            print("Prima apprendi una rete bayesiana per visualizzare le CPT.")
            return

        display(HTML("<h3>Conditional Probability Tables (CPT):</h3>"))
        for node in selected_bayes_net.nodes():
            node_name = selected_bayes_net.variable(node).name()
            display(HTML(f"<h4>Nodo: {node_name}</h4>"))
            cpt = selected_bayes_net.cpt(node_name)
        
            display(cpt)

        display(HTML(f"<h4>Inferenza</h4>"))
        inf = gum.LazyPropagation(selected_bayes_net)
        inf.makeInference()
        gnb.showInference(selected_bayes_net, size=10)

def update_condition_values(change):
    with output:
        output.clear_output()
        condition_var = change.new
        if condition_var in dataset.columns:
            unique_values = list(dataset[condition_var].unique())
            unique_values.insert(0, "")
            dropdown_condition_value.options = unique_values
            dropdown_condition_value.layout = widgets.Layout(width=input_width)
        else:
            dropdown_condition_value.options = [""]

dropdown_condition.observe(update_condition_values, names='value')

def perform_exact_inference(b):
    with output:
        output.clear_output()

        if score_method_dropdown.value == 'Seleziona un metodo':
            print("Errore: prima di procedere con l'inferenza esatta, seleziona un metodo.")
            return

        target = dropdown_target.value
        condition = dropdown_condition.value
        condition_value = dropdown_condition_value.value

        if not target:
            print("❌ Errore: seleziona una variabile target.")
            return

        inf = gum.LazyPropagation(selected_bayes_net)
        inf.makeInference()

        marginal_posterior = inf.posterior(target).topandas()
        display(HTML(f"<h3>🔍 Probabilità Marginale per '{target}'</h3>"))
        if isinstance(marginal_posterior, pd.Series):
            marginal_posterior_df = marginal_posterior.reset_index()
            marginal_posterior_df.columns = list(marginal_posterior_df.columns[:-1]) + ["Probabilità"]
        else:
            marginal_posterior_df = marginal_posterior
        display(HTML(marginal_posterior_df.to_html(index=False, float_format="%.3f")))

        if condition and condition_value:
            try:
                inf.addEvidence(condition, condition_value)
                inf.makeInference()
                conditional_posterior = inf.posterior(target).topandas()

                if isinstance(conditional_posterior, pd.Series):
                    conditional_posterior_df = conditional_posterior.reset_index()
                    conditional_posterior_df.columns = list(conditional_posterior_df.columns[:-1]) + ["Probabilità"]
                else:
                    conditional_posterior_df = conditional_posterior

                display(HTML(f"<h3>🔍 Probabilità Condizionata per '{target}' dato '{condition} = {condition_value}'</h3>"))
                display(HTML(conditional_posterior_df.to_html(index=False, float_format="%.3f")))

                diff_df = conditional_posterior_df.copy()
                diff_df["Probabilità"] -= marginal_posterior_df["Probabilità"]
                
                display(HTML(f"<h3>🔍 Variazione della Probabilità per '{target}' dato '{condition} = {condition_value}'</h3>"))
                display(HTML(diff_df.to_html(index=False, float_format="%.3f")))

                target_column_name = marginal_posterior_df.columns[0]
                max_class = marginal_posterior_df.loc[marginal_posterior_df["Probabilità"].idxmax(), target_column_name]

                marginal_prob = marginal_posterior_df[marginal_posterior_df[target_column_name] == max_class]["Probabilità"].values[0]
                conditional_prob = conditional_posterior_df[conditional_posterior_df[target_column_name] == max_class]["Probabilità"].values[0]
                difference = (conditional_prob - marginal_prob) * 100

            except gum.InvalidArgument as e:
                print(f"❌ Errore nell'aggiunta dell'evidenza: {e}")

        elif condition:
            inf.addJointTarget({target, condition})
            inf.makeInference()
            joint_posterior = inf.jointPosterior({target, condition}).topandas()
            joint_posterior = joint_posterior.div(joint_posterior.sum(axis=0), axis=1)

            display(HTML(f"<h3>Probabilità Congiunta per '{target}' e '{condition}'</h3>"))
            display(HTML(joint_posterior.to_html(index=True, float_format="%.3f")))

def update_dropdown_options():
    variables = selected_bayes_net.names()
    dropdown.options = variables
    dropdown_target.options = variables
    dropdown_condition.options = variables
    from_input.value = ''
    to_input.value = ''

button_check_missing.on_click(display_missing_data)
button_show_distribution.on_click(display_distribution)
button_add_blacklist.on_click(add_to_blacklist)
button_add_whitelist.on_click(add_to_whitelist)
button_learn.on_click(learn_and_display_all_scores)
button_strength.on_click(calculate_arc_strengths)
button_show_cpt.on_click(show_cpt)
button_inference.on_click(perform_exact_inference)

def create_separator():
    return HTML("<hr style='border: none; height: 2px; background-color: #4682B4;' />")

def display_widgets():
    display(
        create_separator(),
        HTML("<h3>Verifica Dati Mancanti:</h3>"),
        button_check_missing,
        output_missing_data,
        create_separator(),
        HTML("<h3>Visualizza Variabili:</h3>"),
        dropdown,
        button_show_distribution,
        output_distribution,
        create_separator(),
        HTML("<h3>Addestra Rete Bayesiana:</h3>"),
        from_input,
        to_input,
        button_add_blacklist,
        button_add_whitelist,
        score_method_dropdown,
        button_learn,
        button_strength,
        button_show_cpt,
        output_bayes_net,
        create_separator(),
        HTML("<h3>Inferenza Esatta:</h3>"),
        dropdown_target,
        dropdown_condition,
        dropdown_condition_value,
        button_inference,
        output_inference,
        create_separator(),
        output
    )

display_widgets()