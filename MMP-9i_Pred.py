import streamlit as st
from streamlit_ketcher import st_ketcher
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors
from rdkit.Chem import PandasTools
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import shap
from PIL import Image
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import io

# Author : Dr. Sk. Abdul Amin
# [Details](https://www.scopus.com/authid/detail.uri?authorId=57190176332).

train_url = "https://github.com/Amincheminform/MMP-9i_Predictor/raw/main/TRAIN12.csv"
test_url = "https://github.com/Amincheminform/MMP-9i_Predictor/raw/main/TEST12.csv"

train_data = pd.read_csv(train_url, sep=',')
test_data = pd.read_csv(test_url, sep=',')

#train_data = pd.read_csv(r'C:\Users\Amin\Downloads\TRAIN12.csv', sep=',')
#test_data = pd.read_csv(r'C:\Users\Amin\Downloads\TEST12.csv', sep=',')

# Streamlit
#logo_url = "https://raw.githubusercontent.com/Amincheminform/phKMOi_v1/main/phKMOi_v1_logo.jpg"

st.set_page_config(
    page_title="MMP-9i Predictor: Predict MMP-9 inhibitors",
    layout="wide",
    #page_icon=logo_url
)

# st.sidebar.image(logo_url)
# st.sidebar.success("Thank you for using KidneyTox_v1.0!")

calc = Calculator(descriptors, ignore_3D=True)
descriptor_columns = ['nS', 'BCUTs-1l', 'NsssN', 'AATS2i', 'BCUTd-1h',
                      'MATS2i', 'EState_VSA5', 'PEOE_VSA3', 'ATSC2i', 'NddssS',
                      'VSA_EState1', 'SlogP_VSA2']

# Train the model
try:
    X_train, y_train = train_data[descriptor_columns], train_data['Binary']
    X_test, y_test = test_data[descriptor_columns], test_data['Binary']

    model = RandomForestClassifier(
        n_estimators=60,
        criterion="entropy",
        max_depth=10,
        min_samples_split=6,
        min_samples_leaf=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    #st.sidebar.success(f"Model trained with test accuracy: {test_accuracy:.2f}")

except Exception as e:
    st.sidebar.error(f"Model training failed: {e}")
    model = None

def generate_2d_image(smiles, img_size=(300, 300)):
    mol = Chem.MolFromSmiles(smiles)
    return Draw.MolToImage(mol, size=img_size, kekulize=True) if mol else None

def mol_to_array(mol, size=(300, 300)):
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    drawer.SetDrawOptions(drawer.drawOptions())  # optionally customize
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    img_data = drawer.GetDrawingText()
    return Image.open(io.BytesIO(img_data))

st.title("MMP-9i Predictor v1.0: Predict MMP-9 inhibitor(s)")

#st.image(logo_url, width=300)

with st.expander("MMP-9i Predictor?", expanded=True):
    st.write('''*MMP-9i Predictor* is a python package that allows users to predict 
             the MMP-9 inhibitory activity of small molecule (**Active** or **Inactive**) 
             and also visualize the molecule.''')

# Two columns: left for sketcher, right for text input
col1, col2 = st.columns(2)

prediction_done = False

with col1:
    st.markdown("### Draw Query Molecule")
    smile_code = st_ketcher()
    if smile_code and not prediction_done:
        st.success("Molecule drawn successfully!")

with col2:
    st.markdown("### SMILES string of Query Molecule")
    smiles_input = st.text_input("Enter or edit SMILES:", value=smile_code if smile_code else "")

    if smiles_input and not prediction_done:
        st.markdown(f"✅ **SMILES code**: `{smiles_input}`")
        #st.markdown("**Calculation may take < 30 seconds!**")
        #st.markdown("**Thank you for your patience!**")

    if smiles_input:
        mol = Chem.MolFromSmiles(smiles_input)
        if mol:
            # Author : Dr. Sk. Abdul Amin
            # [Details](https://www.scopus.com/authid/detail.uri?authorId=57190176332).

            all_data = pd.concat([train_data, test_data], ignore_index=True)

            st.subheader("Results")

            smiles_list = [smiles_input]
            molecules = [Chem.MolFromSmiles(sm) for sm in smiles_list]

            descriptor_df = calc.pandas(molecules)
            external_descriptor_df = descriptor_df[descriptor_columns].dropna()
            X_external = external_descriptor_df

            y_external_pred = model.predict(X_external)

            with st.spinner("Calculating SHAP values..."):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(X_external)

            prediction_done = True

            titles = [
                f"Query molecule\nPredicted Class: {y_external_pred[0]}"
            ]

            # Helper to convert prediction to label
            def pred_label(pred):
                return "### **Active**" if pred == 1 else "### **Inactive**"


            # Row 1 — Query molecule
            #st.markdown("### Query Molecule")
            col1, col2 = st.columns(2)

            with col1:
                # Molecule image + prediction label
                # SHAP plot (smaller)
                plt.figure(figsize=(4, 3))
                shap.plots.waterfall(shap_values[0, :, y_external_pred[0]], max_display=10, show=False)
                fig1 = plt.gcf()
                st.pyplot(fig1)
                plt.clf()
                mol_img = mol_to_array(molecules[0])
                st.image(mol_img, caption="Query Molecule", width=250)
                st.markdown(f"<div style='font-size:40px;'>{pred_label(y_external_pred[0])}</div>",
                            unsafe_allow_html=True)

            with col2:
                # Applicability Domain (Leverage) Plot
                try:
                    X_combined_test = np.vstack((X_train, X_test))
                    Amin_H_test = np.dot(X_combined_test,
                                         np.linalg.pinv(np.dot(X_combined_test.T, X_combined_test))
                                         ).dot(X_combined_test.T)
                    test_leverage = np.diag(Amin_H_test)[len(X_train):]

                    p = X_train.shape[1]
                    n = X_train.shape[0]
                    leverage_threshold = 3 * p / n

                    ad_flags = test_leverage <= leverage_threshold

                    X_external_np = X_external.to_numpy()
                    X_combined_external = np.vstack((X_train, X_external_np))
                    Amin_H_external = np.dot(
                        X_combined_external,
                        np.linalg.pinv(np.dot(X_combined_external.T, X_combined_external))
                    ).dot(X_combined_external.T)
                    external_leverage = np.diag(Amin_H_external)[len(X_train):]
                    external_ad_flags = external_leverage <= leverage_threshold

                    # Plot
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.scatter(
                        range(len(test_leverage)),
                        test_leverage,
                        c=ad_flags,
                        cmap='coolwarm', s=20
                    )
                    ax.scatter(
                        range(len(test_leverage), len(test_leverage) + len(external_leverage)),
                        external_leverage,
                        c=external_ad_flags,
                        cmap='Greens_r',
                        marker='o', s=100,
                        label='Query Molecule'
                    )
                    ax.axhline(y=leverage_threshold, color='red', linestyle='--', label='Threshold')
                    ax.set_xlabel('Compound Index')
                    ax.set_ylabel('Leverage Value')
                    ax.set_title('Applicability Domain Analysis')
                    ax.legend()

                    st.pyplot(fig)
                    st.write("Your Molecule within Applicability Domain:", int(np.sum(external_ad_flags)))
                    st.caption("NOTE: 1 = Yes, 0 = No")
                    st.info("### Don’t forget to cite. Thanks! ###")

                except Exception as e:
                    st.error(f"Applicability Domain analysis failed: {e}")


            # Separator
            st.markdown("---")

    else:
        st.info("Please enter a SMILES string to get predictions.")

# Author : Dr. Sk. Abdul Amin
# [Details](https://www.scopus.com/authid/detail.uri?authorId=57190176332).
# Contact section
with st.expander("Contact", expanded=False):
    st.write('''
        #### Report an Issue

        Report a bug or contribute here: [GitHub](https://github.com/Amincheminform)

        #### Contact Us
        - [Dr. Sk. Abdul Amin](mailto:pharmacist.amin@gmail.com)
    ''')