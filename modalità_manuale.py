import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import pandas as pd
import unicodedata

bayes_net = gum.BayesNet("ReteBayesiana")
gum.config["notebook", "default_arc_color"] = "#4682B4"
gum.config["notebook", "default_node_bgcolor"] = "#4682B4" 
gum.config["notebook", "default_node_fgcolor"] = "#FFFFFF" 
current_cpt_inputs = []

# Output widget
output_dag = widgets.Output()
output_cpt_editor = widgets.Output()
output_log = widgets.Output()
output_cpt_display = widgets.Output()
output_inference = widgets.Output()
output = widgets.Output()


button_width = '200px'
input_width = '300px'

variable_name_input = widgets.Text(
    placeholder="Nome variabile",
    layout=widgets.Layout(width=input_width))
variable_values_input = widgets.Text(
    placeholder="Valori separati da virgola",
    layout=widgets.Layout(width=input_width))
add_variable_button = widgets.Button(
    description="Aggiungi Variabile",
    style={'button_color': '#87CEEB', 'font_weight': 'bold', 'font_color': '#FFFFFF'},
    layout=widgets.Layout(width=button_width))


from_node_input = widgets.Text(placeholder="Da",layout=widgets.Layout(width=input_width))
to_node_input = widgets.Text(placeholder="A",layout=widgets.Layout(width=input_width))
add_arc_button = widgets.Button(
    description="Aggiungi Arco",
    style={'button_color': '#87CEEB', 'font_weight': 'bold', 'font_color': '#FFFFFF'},
    layout=widgets.Layout(width=button_width)
)


select_variable_dropdown = widgets.Dropdown(description="Seleziona Variabile", options=[], style={'description_width': 'initial'} )
save_cpt_button = widgets.Button(description="Salva CPT", style={'button_color': '#87CEEB', 'font_weight': 'bold', 'font_color': '#FFFFFF'},
    layout=widgets.Layout(width=button_width))
show_cpt_button = widgets.Button(description="Mostra CPT",style={'button_color': '#87CEEB', 'font_weight': 'bold', 'font_color': '#FFFFFF'},
    layout=widgets.Layout(width=button_width))

dropdown_target = widgets.Dropdown(
    description='Evento:',
    options=[],
    style={'description_width': 'initial'}
)

dropdown_condition = widgets.Dropdown(
    description='Evidenza:',
    options=[],
    style={'description_width': 'initial'}
)

dropdown_condition_value = widgets.Dropdown(
    description='Valore:',
    style={'description_width': 'initial'}
)

button_inference = widgets.Button(
    description="Esegui Inferenza",
    style={'button_color': '#87CEEB', 'font_weight': 'bold', 'font_color': '#FFFFFF'},
    layout=widgets.Layout(width=button_width)
)

view_dag_button = widgets.Button(
    description="Visualizza DAG",
    style={'button_color': '#87CEEB', 'font_weight': 'bold', 'font_color': '#FFFFFF'},
    layout=widgets.Layout(width=button_width)
)

def normalize_text(text):
    """Rimuove accenti e caratteri speciali dai nomi delle variabili."""
    text = text.strip().replace(" ", "_")  
    text = ''.join(c for c in unicodedata.normalize('NFKD', text) if unicodedata.category(c) != 'Mn') 
    return text

def add_variable(b):
    """Aggiunge variabile alla rete bayesiana con nomi e valori normalizzati."""
    with output_log:
        clear_output()
        name = normalize_text(variable_name_input.value)  
        values = [normalize_text(v) for v in variable_values_input.value.split(',')] 
        
        if not name or not values:
            print("⚠️ Errore: Nome e valori della variabile sono obbligatori.")
            return
        
        if name in bayes_net.names():
            print(f"⚠️ Errore: Variabile duplicata: Impossibile inserire la variabile con il nome '{name}'.")
            return
        
        try:
            new_var = gum.LabelizedVariable(name, name, len(values))
            for i, val in enumerate(values):
                new_var.changeLabel(i, val)
            bayes_net.add(new_var)
            variable_values_dict[name] = values  # Memorizza i valori
            print(f"✅ Variabile '{name}' aggiunta con valori: {values}")
            update_variable_dropdown()
            update_variable_dropdown_target()
            update_variable_dropdown_conditions()
        except Exception as e:
            print(f"❌ Errore: {e}")

variable_values_dict = {}

def add_arc(b):
    """Aggiunge arco tra due variabili."""
    with output_dag:
        clear_output()
        from_node = from_node_input.value.strip()
        to_node = to_node_input.value.strip()
        
        if not from_node or not to_node:
            print("⚠️ Errore: Specifica i nodi 'da' e 'a'.")
            return
        
        try:
            bayes_net.addArc(from_node, to_node)
            print(f"✅ Arco aggiunto: {from_node} -> {to_node}")
            gnb.showBN(bayes_net)
        except Exception as e:
            print(f"❌ Errore: Variabile non trovata")

def update_variable_dropdown():
    """Aggiorna il dropdown delle variabili."""
    select_variable_dropdown.options = bayes_net.names()

def update_variable_dropdown_target():
    dropdown_target.options = bayes_net.names()

def update_variable_dropdown_conditions():
    dropdown_condition.options = bayes_net.names()

def generate_cpt_inputs(variable_name):
    """Genera i widget di input per le probabilità condizionate."""
    global current_cpt_inputs
    current_cpt_inputs = []
    cpt = bayes_net.cpt(variable_name)
    inst = gum.Instantiation(cpt)
    inst.setFirst()
    
    input_widgets = []
    while not inst.end():

        states = {inst.variable(i).name(): inst.variable(i).label(inst.val(i)) 
                  for i in range(inst.nbrDim())}
        state_desc = " , ".join([f"{k}={v}" for k, v in states.items()])
          
        select_variable = widgets.Label(value=f"P({variable_name} | {state_desc})")
        prob_input = widgets.FloatText(value=0.0, layout=widgets.Layout(width='200px'))
        
        input_row = widgets.HBox([select_variable, prob_input])
        input_widgets.append(input_row)
        
        current_cpt_inputs.append((prob_input, {'cpt': cpt, 'states': states}))
        inst.inc()
    
    with output_cpt_editor:
        clear_output()
        display(widgets.VBox(input_widgets), save_cpt_button)

def save_cpt(b):
    """Salva i valori inseriti nelle CPT."""
    with output_cpt_editor:
        clear_output()
        try:
            probability_sums = {}  
            target_variable = select_variable_dropdown.value
            
            parent_variables = [bayes_net.variable(parent).name() for parent in bayes_net.parents(target_variable)]

            for widget, data in current_cpt_inputs:
                cpt = data['cpt']
                states = data['states']
                inst = gum.Instantiation(cpt)

                for var_name, state in states.items():
                    inst.chgVal(var_name, state)

                prob_value = widget.value
                parent_state_values = tuple(states[parent] for parent in parent_variables)
                if parent_state_values not in probability_sums:
                    probability_sums[parent_state_values] = 0
                probability_sums[parent_state_values] += prob_value

                cpt.set(inst, prob_value)

            for state, total in probability_sums.items():
                if not (abs(total - 1.0) < 1e-6):
                    print(f"❌ Errore: La somma delle probabilità per {target_variable} non è pari a 1. Somma attuale: {total:.2f}")
                    return

            print("✅ CPT salvata con successo!")
        except Exception as e:
            print(f"❌ Errore durante il salvataggio della CPT: {e}")


def show_cpt(b):
    """Mostra la CPT generale della rete bayesiana."""
    with output_cpt_display:
        clear_output()
        try:
            print("🔍 Mostra CPT:")
            for variable_name in bayes_net.names():
                print(f"🔍 CPT di '{variable_name}':")
                gnb.showPotential(bayes_net.cpt(variable_name))
        except Exception as e:
            print(f"❌ Errore: {e}")

def perform_exact_inference(b):
    with output_inference:
        output_inference.clear_output()

        target = dropdown_target.value
        condition = dropdown_condition.value
        condition_value = dropdown_condition_value.value  

        if not target:
            print("❌ Errore: seleziona una variabile target.")
            return
        inf = gum.LazyPropagation(bayes_net)  
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
            except gum.InvalidArgument as e:
                print(f"❌ Errore nell'aggiunta dell'evidenza: {e}")
        elif condition:
       
            inf.addJointTarget({target, condition})
            inf.makeInference()
            joint_posterior = inf.jointPosterior({target, condition}).topandas()
            joint_posterior = joint_posterior.div(joint_posterior.sum(axis=0), axis=1)

            display(HTML(f"<h3>🔍 Probabilità Congiunta per '{target}' e '{condition}'</h3>"))
            display(HTML(joint_posterior.to_html(index=True, float_format="%.3f")))

def update_condition_values(change):
    with output:
        output.clear_output()
        condition_var = change.new 
        print(f"Variabile condizionante selezionata: {condition_var}") 
        if condition_var in variable_values_dict:  
            unique_values = variable_values_dict[condition_var]  
            print(f"Valori trovati: {unique_values}")  
            unique_values.insert(0, "") 
            dropdown_condition_value.options = unique_values
            dropdown_condition_value.layout = widgets.Layout(width=input_width) 
        else:
            print("Nessun valore trovato per la variabile condizionante.")  
            dropdown_condition_value.options = []
dropdown_condition.observe(update_condition_values, names='value')

def view_dag(b):
    """Visualizza il DAG della rete bayesiana."""
    with output_dag:
        clear_output()
        gnb.showBN(bayes_net, size="800")

add_variable_button.on_click(add_variable)
add_arc_button.on_click(add_arc)
select_variable_dropdown.observe(lambda change: generate_cpt_inputs(change.new), names='value')
save_cpt_button.on_click(save_cpt)
show_cpt_button.on_click(show_cpt)
button_inference.on_click(perform_exact_inference)
view_dag_button.on_click(view_dag)

file_name_input = widgets.Text(placeholder="Nome file (senza estensione)", layout=widgets.Layout(width=input_width))
save_button = widgets.Button(description="Salva Rete", style={'button_color': '#87CEEB'}, layout=widgets.Layout(width=button_width))
load_button = widgets.Button(description="Carica Rete", style={'button_color': '#87CEEB'}, layout=widgets.Layout(width=button_width))
output_save_load = widgets.Output()

def save_bayes_net(b):
    """Salva la rete bayesiana in un file .bif"""
    with output_save_load:
        clear_output()
        file_name = file_name_input.value.strip()
        if not file_name:
            print("⚠️ Errore: Specifica un nome per il file.")
            return
        
        file_path = f"{file_name}.bif"
        try:
            gum.saveBN(bayes_net, file_path)
            print(f"✅ Rete salvata con successo in '{file_path}'")
        except Exception as e:
            print(f"❌ Errore nel salvataggio: {e}")

def load_bayes_net(b):
    """Carica una rete bayesiana da un file .bif e aggiorna i widget."""
    with output_save_load:
        clear_output()
        file_name = file_name_input.value.strip()
        if not file_name:
            print("⚠️ Errore: Specifica un nome per il file.")
            return
        
        file_path = f"{file_name}.bif"
        global bayes_net
        try:
            bayes_net = gum.loadBN(file_path)
            print(f"✅ Rete caricata con successo da '{file_path}'")
            update_variable_dropdown()
            update_variable_dropdown_target()
            update_variable_dropdown_conditions()
            gnb.showBN(bayes_net)
            show_cpt(None)

        except Exception as e:
            print(f"❌ Errore nel caricamento: {e}")


save_button.on_click(save_bayes_net)
load_button.on_click(load_bayes_net)

def create_separator():
    return HTML("<hr style='border: none; height: 2px; background-color: #4682B4;' />")

def display_widgets():
    display(
        create_separator(),
        HTML("<h3>Creazione Variabili</h3>"),
        variable_name_input,
        variable_values_input,
        add_variable_button,
        output_log,
        create_separator(),
        HTML("<h3>Crea Arco:</h3>"),
        from_node_input,
        to_node_input,
        add_arc_button,
        output_dag,
        view_dag_button,
        create_separator(),
        HTML("<h3>Crea CPT:</h3>"),
        select_variable_dropdown,
        output_cpt_editor,
        create_separator(),
        HTML("<h3>Visualizzazione CPT</h3>"),
        show_cpt_button,
        output_cpt_display,
        create_separator(),
        HTML("<h3>Inferenza Esatta:</h3>"),
        dropdown_target,
        dropdown_condition,
        dropdown_condition_value,
        button_inference,
        output_inference,
        create_separator(),
        HTML("<h3>Salvataggio e Caricamento:</h3>"),
        file_name_input,
        save_button,
        load_button,
        output_save_load,
        create_separator()
    )

display_widgets()
