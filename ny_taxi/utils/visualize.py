import os
import numpy as np
import streamlit as st


from ny_taxi.config.config import DataLoaderConfig
from ny_taxi.dataset.data_loader import data_loader
from ny_taxi.dataset.data_transformer import transform
from ny_taxi.utils.plot import create_bar_chart, create_pie_chart


def visualize() -> None:
    dir_dataset = st.sidebar.text_input("Select dataset dir", "dataset_ny_taxi")
    year = st.sidebar.selectbox("Select year", [2021, 2024])
    month = st.sidebar.selectbox("Select month", np.arange(13))
    taxi_type = st.sidebar.selectbox(
        "Select taxi type", ["green", "yellow", "fhv", "fhvhv"]
    )

    config_dataloader = DataLoaderConfig(
        dir_dataset=dir_dataset, year=year, taxi_type=taxi_type, month=month
    )
    df_ny_taxi = data_loader(config_dataloader)
    df_ny_taxi = transform(df_ny_taxi)
    st.write(config_dataloader.all_files)
    # st.write(df_ny_taxi.shape)
    st.write("For categorical features, code: 1000 is for unknown (missing)")
    st.dataframe(df_ny_taxi)

    df_counts_vendor_ids = (
        df_ny_taxi.value_counts("VendorID", sort=True)
        .rename_axis("VendorID")
        .reset_index(name="counts")
    )
    fig_1 = create_pie_chart(
        df_counts_vendor_ids.counts.to_numpy(),
        df_counts_vendor_ids.VendorID.to_list(),
        f"Distribution of Vendors for taxi:{taxi_type} trips, year:{year}, month:{month}",
        "Vendor",
    )
    st.pyplot(fig_1)
    st.write(df_counts_vendor_ids)

    df_counts_payment_type = (
        df_ny_taxi.value_counts("payment_type", sort=True)
        .rename_axis("payment_type")
        .reset_index(name="counts")
    )
    fig_2 = create_pie_chart(
        df_counts_payment_type.counts.to_numpy(),
        df_counts_payment_type.payment_type.astype(np.int32).to_list(),
        f"Distribution of Payment type for taxi:{taxi_type} trips, year:{year}, month:{month}",
        "Payment_Type",
    )
    st.pyplot(fig_2)
    st.write(df_counts_payment_type)

    df_counts_trip_type = (
        df_ny_taxi.value_counts("trip_type", sort=True)
        .rename_axis("trip_type")
        .reset_index(name="counts")
    )
    fig_3 = create_pie_chart(
        df_counts_trip_type.counts.to_numpy(),
        df_counts_trip_type.trip_type.astype(np.int32).to_list(),
        f"Distribution of Trip type for taxi:{taxi_type} trips, year:{year}, month:{month}",
        "Trip_Type",
    )
    st.pyplot(fig_3)
    st.write(df_counts_trip_type)

    num_locs_min = 5
    num_locs_max = 100
    top_N_locations = st.number_input(
        f"Select top N locations for visualization ({num_locs_min}-{num_locs_max})",
        min_value=num_locs_min,
        max_value=num_locs_max,
        value="min",
    )
    df_counts_PU_location = (
        df_ny_taxi.value_counts("PULocationID", sort=True)
        .rename_axis("PULocationID")
        .reset_index(name="counts")
    )
    fig_4 = create_bar_chart(
        df_counts_PU_location.counts.to_numpy()[:top_N_locations],
        df_counts_PU_location.PULocationID.astype(np.int32).to_list()[:top_N_locations],
        f"Top {top_N_locations} Pickup locations for taxi:{taxi_type} trips, year:{year}, month:{month}",
    )
    st.pyplot(fig_4)
    st.write(df_counts_PU_location)

    df_counts_DO_location = (
        df_ny_taxi.value_counts("DOLocationID", sort=True)
        .rename_axis("DOLocationID")
        .reset_index(name="counts")
    )
    fig_5 = create_bar_chart(
        df_counts_DO_location.counts.to_numpy()[:top_N_locations],
        df_counts_DO_location.DOLocationID.astype(np.int32).to_list()[:top_N_locations],
        f"Top {top_N_locations} Dropoff locations for taxi:{taxi_type} trips, year:{year}, month:{month}",
    )
    st.pyplot(fig_5)
    st.write(df_counts_DO_location)

    df_counts_passenger_count = (
        df_ny_taxi.value_counts("passenger_count", sort=True)
        .rename_axis("passenger_count")
        .reset_index(name="counts")
    )
    fig_6 = create_pie_chart(
        df_counts_passenger_count.counts.to_numpy(),
        df_counts_passenger_count.passenger_count.astype(np.int32).to_list(),
        f"Distribution of Passenger counts for taxi:{taxi_type} trips, year:{year}, month:{month}",
        "Passenger_Counts",
    )
    st.pyplot(fig_6)
    st.write(df_counts_passenger_count)

    return
