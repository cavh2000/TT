from flask import Flask, render_template, request
from flask import (
    Flask,
    render_template,
    request,
    send_file
)

import geopandas as gpd
import numpy as np
import pandas as pd
import folium
import branca.colormap as cm
import joblib
import torch
import torch.nn.functional as F

from tensorflow.keras.models import load_model

from torch_geometric.nn import GCNConv

from sklearn.neighbors import kneighbors_graph

from branca.element import Template, MacroElement

from torch.nn import BatchNorm1d

from torch_geometric.data import Data


# =====================================================
# FLASK
# =====================================================

app = Flask(__name__)

# =====================================================
# NORMALIZACIONES
# =====================================================

def normalizar_log(x):

    x = np.log1p(x)

    return (
        (x - x.min())
        /
        (x.max() - x.min() + 1e-9)
    )

def normalizar_quantiles(x):

    return (
        pd.qcut(
            x.rank(method='first'),
            q=10,
            labels=False
        ) / 9
    )

# =====================================================
# GNN
# =====================================================

from torch_geometric.nn import (
    SAGEConv,
    BatchNorm
)

class GraphSAGEModel(torch.nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.sage1 = SAGEConv(
            input_dim,
            128
        )

        self.bn1 = BatchNorm(128)

        self.sage2 = SAGEConv(
            128,
            64
        )

        self.bn2 = BatchNorm(64)

        self.sage3 = SAGEConv(
            64,
            64
        )

        self.bn3 = BatchNorm(64)

        self.fc1 = torch.nn.Linear(
            64,
            64
        )

        self.bn_fc1 = BatchNorm(64)

        self.fc2 = torch.nn.Linear(
            64,
            32
        )

        self.bn_fc2 = BatchNorm(32)

        self.out = torch.nn.Linear(
            32,
            2
        )

        self.dropout = 0.2

    def forward(self, data):

        x = data.x
        edge_index = data.edge_index

        # GraphSAGE 1
        x1 = self.sage1(
            x,
            edge_index
        )

        x1 = self.bn1(x1)

        x1 = F.relu(x1)

        x1 = F.dropout(
            x1,
            p=self.dropout,
            training=self.training
        )

        # GraphSAGE 2
        x2 = self.sage2(
            x1,
            edge_index
        )

        x2 = self.bn2(x2)

        x2 = F.relu(x2)

        x2 = F.dropout(
            x2,
            p=self.dropout,
            training=self.training
        )

        residual = x2

        # GraphSAGE 3
        x3 = self.sage3(
            x2,
            edge_index
        )

        x3 = self.bn3(x3)

        x3 = F.relu(x3)

        x3 = F.dropout(
            x3,
            p=self.dropout,
            training=self.training
        )

        x3 = x3 + residual

        # Dense 64
        x3 = self.fc1(x3)

        x3 = self.bn_fc1(x3)

        x3 = F.relu(x3)

        x3 = F.dropout(
            x3,
            p=self.dropout,
            training=self.training
        )

        # Dense 32
        x3 = self.fc2(x3)

        x3 = self.bn_fc2(x3)

        x3 = F.relu(x3)

        x3 = F.dropout(
            x3,
            p=self.dropout,
            training=self.training
        )

        return self.out(x3)

# =====================================================
# LEYENDA
# =====================================================

def agregar_leyenda(mapa):

    template = """
    {% macro html(this, kwargs) %}

    <div style="
        position: fixed;
        bottom: 20px;
        right: 10px;
        z-index: 9999;

        background-color: white;

        padding: 12px;

        border: 2px solid grey;
        border-radius: 8px;

        box-shadow: 2px 2px 8px rgba(0,0,0,0.3);

        font-size: 12px;

        width: 230px;

        opacity: 0.95;
    ">

    <h4>Escalas</h4>

    <b>Delitos</b><br>

    <div style="
        width:200px;
        height:12px;
        background:
        linear-gradient(
            to right,
            #fff5f0,
            #67000d
        );
    ">
    </div>

    <br>

    <b>Población</b><br>

    <div style="
        width:200px;
        height:12px;
        background:
        linear-gradient(
            to right,
            #fcfbfd,
            #3f007d
        );
    ">
    </div>

    <br>

    <b>Probabilidad</b><br>

    <div style="
        width:200px;
        height:12px;
        background:
        linear-gradient(
            to right,
            #f7fcf5,
            #00441b
        );
    ">
    </div>

    <br>

    <b>Superposición</b><br>

    <div style="
        width:200px;
        height:12px;
        background:
        linear-gradient(
            to right,
            #fff5eb,
            #7f2704
        );
    ">
    </div>

    </div>

    {% endmacro %}
    """

    macro = MacroElement()

    macro._template = Template(template)

    mapa.get_root().add_child(macro)

# =====================================================
# DASHBOARD
# =====================================================

def agregar_dashboard(
    mapa,
    num_existentes,
    num_propuestas,
    cobertura,
    dist,
    redundancia,
    no_cubiertas,
    red,
    prob_promedio,
    score_promedio
):

    html = f"""
    {{% macro html(this, kwargs) %}}

    <div style="
        position: fixed;

        bottom: 20px;
        left: 20px;

        width: 270px;

        z-index:9999;

        background:white;

        padding:15px;

        border-radius:10px;

        border:2px solid gray;

        box-shadow:
        2px 2px 10px rgba(0,0,0,0.3);

        font-size:14px;

        opacity:0.95;
    ">

    <h3>Dashboard</h3>

    <b>Red utilizada:</b><br>
    {red}

    <br><br>

    <b>Cámaras existentes:</b><br>
    {num_existentes}

    <br><br>

    <b>Cámaras propuestas:</b><br>
    {num_propuestas}

    <br><br>

    <b>Cobertura estimada:</b><br>
    {cobertura:.2f}%

    <br><br>

    <b>Radio cobertura:</b><br>
    {dist} m

    <br><br>

    <b>Casillas redundantes:</b><br>
    {redundancia}

    <br><br>

    <b>Zonas no cubiertas:</b><br>
    {no_cubiertas}

    <br><br>

    <b>Probabilidad promedio:</b><br>
    {prob_promedio:.3f}

    <br><br>

    <b>Score promedio:</b><br>
    {score_promedio:.3f}

    <br><br>

    </div>

    {{% endmacro %}}
    """

    macro = MacroElement()

    macro._template = Template(html)

    mapa.get_root().add_child(macro)

# =====================================================
# HOME
# =====================================================

@app.route("/", methods=["GET", "POST"])
def index():

    mapa_html = None

    if request.method == "POST":

        # =====================================================
        # PARÁMETROS
        # =====================================================

        RED = request.form["red"]

        MAX_CAMARAS = int(
            request.form["num"]
        )

        DIST_NUEVAS = float(
            request.form["dist_nuevas"]
        )

        DIST_EXISTENTES = float(
            request.form["dist_existentes"]
        )

        BUFFER_COBERTURA = float(
            request.form["buffer_cobertura"]
        )

        # =====================================================
        # CARGA
        # =====================================================

        grid = gpd.read_file(
            "GRID_GAM_DATASET_FINAL.geojson"
        )

        camaras_80 = gpd.read_file(
            "camaras_80.geojson"
        )

        camaras_20 = gpd.read_file(
            "camaras_20.geojson"
        )

        # =====================================================
        # MODELO SEGÚN RED
        # =====================================================

        if RED == "MLP":

            model = load_model(
                "modelo_mlp.h5"
            )

            scaler = joblib.load(
                "scaler_mlp.pkl"
            )

            feature_names = joblib.load(
                "features_mlp.pkl"
            )

        else:

            scaler = joblib.load(
                "scaler_gnn.pkl"
            )

            feature_names = joblib.load(
                "features_gnn.pkl"
            )

        # =====================================================
        # FEATURES
        # =====================================================

        features = grid[
            feature_names
        ].fillna(0)

        X_scaled = scaler.transform(
            features
        )

        # =====================================================
        # PREDICCIÓN
        # =====================================================

        if RED == "MLP":

            grid["probabilidad"] = (
                model.predict(X_scaled)
                .flatten()
                .astype(float)
            )

        else:

            # =====================================================
            # GRAFO ESPACIAL
            # =====================================================

            grid_proj = grid.to_crs(
                epsg=3857
            )

            centroids = np.array([

                [
                    geom.centroid.x,
                    geom.centroid.y
                ]

                for geom in grid_proj.geometry
            ])

            A = kneighbors_graph(
                centroids,
                n_neighbors=8,
                mode='connectivity',
                include_self=False
            )

            edge_index = np.array(
                A.nonzero()
            )

            edge_index = torch.tensor(
                edge_index,
                dtype=torch.long
            )

            # =====================================================
            # TENSOR FEATURES
            # =====================================================

            x = torch.tensor(
                X_scaled,
                dtype=torch.float
            )

            data = Data(
                x=x,
                edge_index=edge_index
            )

            # =====================================================
            # MODELO GNN
            # =====================================================

            model = GraphSAGEModel(
                input_dim=x.shape[1]
            )

            model.load_state_dict(
                torch.load(
                "modelo_gnn.pth",
                map_location=torch.device('cpu')
            )
)

            model.eval()

            # =====================================================
            # INFERENCIA
            # =====================================================

            with torch.no_grad():

                out = model(data)

                probs = F.softmax(
                    out,
                    dim=1
                )[:,1]

            grid["probabilidad"] = (
                probs.cpu().numpy()
            )

        # =====================================================
        # NORMALIZACIÓN
        # =====================================================

        grid["delitos_norm"] = (
            normalizar_log(
                grid["delitos_totales"]
            )
        )

        grid["poblacion_norm"] = (
            normalizar_quantiles(
                grid["poblacion_proporcional"]
            )
        )

        grid["prob_norm"] = (
            normalizar_log(
                grid["probabilidad"]
            )
        )

        grid["delitos_score"] = (
            grid["delitos_totales"]
            /
            (
                grid["delitos_totales"].max()
                + 1e-9
            )
        )

        grid["poblacion_score"] = (
            grid["poblacion_proporcional"]
            /
            (
                grid["poblacion_proporcional"].max()
                + 1e-9
            )
        )

        grid["transporte_score"] = (
            1
            /
            (
                grid["dist_transporte_min"]
                + 1
            )
        )

        grid["transporte_score"] = (
            grid["transporte_score"]
            /
            (
                grid["transporte_score"].max()
                + 1e-9
            )
        )

        grid["score_final"] = (

            grid["probabilidad"] * 0.50 +

            grid["delitos_score"] * 0.30 +

            grid["poblacion_score"] * 0.15 +

            grid["transporte_score"] * 0.05
        )

        # =====================================================
        # CRS MÉTRICO
        # =====================================================

        if grid.crs.is_geographic:

            grid = grid.to_crs(
                epsg=3857
            )

        camaras_80 = (
            camaras_80.to_crs(
                grid.crs
            )
        )

        camaras_20 = (
            camaras_20.to_crs(
                grid.crs
            )
        )

        # =====================================================
        # PROPUESTAS
        # =====================================================

        grid["centroid"] = (
            grid.geometry.centroid
        )

        grid_sorted = (
            grid.sort_values(
                by="score_final",
                ascending=False
            )
        )

        seleccionados = []

        buffer_existentes = (
            camaras_80.buffer(
                DIST_EXISTENTES
            )
        )

        zona_cubierta = (
            buffer_existentes.unary_union
        )

        for _, row in grid_sorted.iterrows():

            punto = row["centroid"]

            if punto.within(
                zona_cubierta
            ):
                continue

            aceptar = True

            for sel in seleccionados:

                if (
                    punto.distance(sel)
                    < DIST_NUEVAS
                ):
                    aceptar = False
                    break

            if aceptar:
                seleccionados.append(
                    punto
                )

            if (
                len(seleccionados)
                >= MAX_CAMARAS
            ):
                break

        nuevas = gpd.GeoDataFrame(
            geometry=seleccionados,
            crs=grid.crs
        )

        # guardar propuestas

        nuevas_export = nuevas.copy()

        nuevas_export.to_crs(
            epsg=4326
        ).to_file(
            "camaras_propuestas.geojson",
            driver="GeoJSON"
        )

        # =====================================================
        # COBERTURA
        # =====================================================

        buffers_existentes = (
            camaras_80.buffer(
                BUFFER_COBERTURA
            )
        )

        buffers_nuevas = (
            nuevas.buffer(
                BUFFER_COBERTURA
            )
        )

        # =====================================================
        # COBERTURA TOTAL
        # =====================================================

        union_existentes = (
            buffers_existentes.unary_union
        )

        union_nuevas = (
            buffers_nuevas.unary_union
        )

        union_total = (
            union_existentes.union(
                union_nuevas
            )
        )

        # =====================================================
        # % DE CELDAS CUBIERTAS
        # =====================================================

        grid["cubierta"] = (
            grid.geometry.intersects(
                union_total
            )
        )

        cobertura = (
            grid["cubierta"].sum()
            /
            len(grid)
        ) * 100

        # =====================================================
        # SUPERPOSICIÓN
        # =====================================================

        todos_buffers = (
            list(buffers_existentes)
            +
            list(buffers_nuevas)
        )

        grid["superposicion"] = 0

        for idx, row in grid.iterrows():

            celda = row.geometry

            contador = 0

            for buffer in todos_buffers:

                if celda.intersects(buffer):
                    contador += 1

            grid.at[
                idx,
                "superposicion"
            ] = contador

        grid["super_norm"] = (
            normalizar_log(
                grid["superposicion"]
            )
        )

        # =====================================================
        # ZONAS NO CUBIERTAS
        # =====================================================

        redundancia = len(
            grid[
                grid["superposicion"] > 2
            ]
        )

        no_cubiertas_count = len(
            grid[
                ~grid["cubierta"]
            ]
        )

        # =====================================================
        # LAT/LON
        # =====================================================

        grid = grid.to_crs(
            epsg=4326
        )

        camaras_80 = (
            camaras_80.to_crs(
                epsg=4326
            )
        )

        camaras_20 = (
            camaras_20.to_crs(
                epsg=4326
            )
        )

        nuevas = nuevas.to_crs(
            epsg=4326
        )

        buffers_existentes = buffers_existentes.to_crs(
            epsg=4326
        )

        buffers_nuevas = buffers_nuevas.to_crs(
            epsg=4326
        )


        # =====================================================
        # MAPA
        # =====================================================

        center = grid.unary_union.centroid

        m = folium.Map(
            location=[
                center.y,
                center.x
            ],
            zoom_start=12
        )

        # =====================================================
        # COLORMAPS
        # =====================================================

        cmap_delitos = (
            cm.linear.Reds_09.scale(0, 1)
        )

        cmap_poblacion = (
            cm.linear.Purples_09.scale(0, 1)
        )

        cmap_prob = (
            cm.linear.Greens_09.scale(0, 1)
        )

        cmap_super = (
            cm.linear.Oranges_09.scale(0, 1)
        )

        # =====================================================
        # CAPAS GRID
        # =====================================================

        def pintar_capa(
            nombre,
            columna,
            cmap,
            visible
        ):

            fg = folium.FeatureGroup(
                name=nombre,
                show=visible
            )

            for _, row in grid.iterrows():

                val = row[columna]

                folium.GeoJson(
                    row.geometry,

                    style_function=lambda x, v=val: {
                        "fillColor": cmap(v),
                        "color": "black",
                        "weight": 0.2,
                        "fillOpacity": 0.7
                    }

                ).add_to(fg)

            fg.add_to(m)

        pintar_capa(
            "Delitos",
            "delitos_norm",
            cmap_delitos,
            True
        )

        pintar_capa(
            "Población",
            "poblacion_norm",
            cmap_poblacion,
            False
        )

        pintar_capa(
            "Probabilidad",
            "prob_norm",
            cmap_prob,
            False
        )

        pintar_capa(
            "Superposición",
            "super_norm",
            cmap_super,
            False
        )

        # =====================================================
        # GRID
        # =====================================================

        fg_grid = folium.FeatureGroup(
            name="Grid",
            show=True
        )

        folium.GeoJson(
            grid.geometry,

            style_function=lambda x: {
                "fillOpacity": 0,
                "color": "gray",
                "weight": 0.3
            }

        ).add_to(fg_grid)

        fg_grid.add_to(m)

        # =====================================================
        # CÁMARAS 2024
        # =====================================================

        fg_2024 = folium.FeatureGroup(
            name="Cámaras 80%",
            show=True
        )

        for _, row in camaras_80.iterrows():

            folium.CircleMarker(
                [
                    row.geometry.y,
                    row.geometry.x
                ],

                radius=3,
                color="blue",
                fill=True

            ).add_to(fg_2024)

        fg_2024.add_to(m)

        # =====================================================
        # CÁMARAS 20%
        # =====================================================

        fg_20 = folium.FeatureGroup(
            name="Cámaras 20%",
            show=True
        )

        for _, row in camaras_20.iterrows():

            folium.CircleMarker(
                [
                    row.geometry.y,
                    row.geometry.x
                ],

                radius=4,
                color="white",
                fill=True,
                fill_color="white"

            ).add_to(fg_20)

        fg_20.add_to(m)

        # =====================================================
        # PROPUESTAS
        # =====================================================

        fg_prop = folium.FeatureGroup(
            name="Cámaras propuestas",
            show=True
        )

        for _, row in nuevas.iterrows():

            folium.CircleMarker(
                [
                    row.geometry.y,
                    row.geometry.x
                ],

                radius=4,
                color="black",
                fill=True

            ).add_to(fg_prop)

        fg_prop.add_to(m)

        # =====================================================
        # COBERTURA EXISTENTE
        # =====================================================

        fg_cov_exist = (
            folium.FeatureGroup(
                name="Cobertura cámaras 80%",
                show=False
            )
        )

        for geom in buffers_existentes:

            folium.GeoJson(
                geom,

                style_function=lambda x: {
                    "fillColor": "blue",
                    "color": "blue",
                    "weight": 1,
                    "fillOpacity": 0.08
                }

            ).add_to(fg_cov_exist)

        fg_cov_exist.add_to(m)

        # =====================================================
        # COBERTURA PROPUESTAS
        # =====================================================

        fg_cov_prop = (
            folium.FeatureGroup(
                name="Cobertura propuestas",
                show=True
            )
        )

        for geom in buffers_nuevas:

            folium.GeoJson(
                geom,

                style_function=lambda x: {
                    "fillColor": "black",
                    "color": "black",
                    "weight": 1,
                    "fillOpacity": 0.08
                }

            ).add_to(fg_cov_prop)

        fg_cov_prop.add_to(m)

        # =====================================================
        # ZONAS NO CUBIERTAS
        # =====================================================

        fg_no_cubiertas = (
            folium.FeatureGroup(
                name="Zonas no cubiertas",
                show=False
            )
        )

        no_cubiertas = grid[
            grid["cubierta"] == False
        ]

        for _, row in no_cubiertas.iterrows():

            folium.GeoJson(
                row.geometry,

                style_function=lambda x: {
                    "fillColor": "white",
                    "color": "black",
                    "weight": 0.5,
                    "fillOpacity": 0.8
                }

            ).add_to(fg_no_cubiertas)

        fg_no_cubiertas.add_to(m)

        # =====================================================
        # CONTROLES
        # =====================================================

        folium.LayerControl(
            collapsed=False
        ).add_to(m)

        agregar_leyenda(m)

        prob_promedio = (
            grid["probabilidad"]
            .mean()
        )

        score_promedio = (
            grid["score_final"]
            .mean()
        )

        agregar_dashboard(
            m,
            len(camaras_80),
            len(nuevas),
            cobertura,
            BUFFER_COBERTURA,
            redundancia,
            no_cubiertas_count,
            RED,
            prob_promedio,
            score_promedio
        )

        mapa_html = m._repr_html_()

    return render_template(
        "index.html",
        mapa=mapa_html
    )

# ==========================================
# DESCARGAR GEOJSON
# ==========================================

@app.route("/exportar_geojson")
def exportar_geojson():

    return send_file(
        "camaras_propuestas.geojson",
        as_attachment=True,
        download_name="camaras_propuestas.geojson"
    )


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":

    app.run(debug=True)
