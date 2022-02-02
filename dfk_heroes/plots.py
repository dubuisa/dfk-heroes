import altair as alt

def advanced_analytics(df_cv, width=500, height=250):

    brush = alt.selection(type='interval',resolve='global')

    points = alt.Chart(df_cv).mark_point().encode(
        x='TSNE-1',
        y='TSNE-2',
        #color=alt.condition(brush, 'Predicted soldPrice (Quintile)', alt.value('lightgray')),
        color=alt.Color('Predicted soldPrice (Quintile)',scale=alt.Scale(scheme='redyellowgreen'))
    ).add_selection(
        brush
    ).properties(
            width=width,
            height=height
    )


    bars_quantile = alt.Chart(df_cv).mark_bar().encode(
        y='Predicted soldPrice (Quintile)',
        color=alt.Color('Predicted soldPrice (Quintile)'),
        x='count(Predicted soldPrice (Quintile))'
    ).transform_filter(
        brush
    ).properties(
            width=width,
            height=height
    )

    bars_profession = alt.Chart(df_cv).mark_bar().encode(
        y='profession',
        color='Predicted soldPrice (Quintile)',
        x='count(Predicted soldPrice (Quintile))'
    ).transform_filter(
        brush
    ).properties(
            width=width,
            height=height
    )

    bars_rarity = alt.Chart(df_cv).mark_bar().encode(
        y='rarity',
        color='Predicted soldPrice (Quintile)',
        x='count(Predicted soldPrice (Quintile))'
    ).transform_filter(
        brush
    ).properties(
            width=width,
            height=height
    )

    bars_mainclass = alt.Chart(df_cv).mark_bar().encode(
        y='mainClass',
        color='Predicted soldPrice (Quintile)',
        x='count(Predicted soldPrice (Quintile))'
    ).transform_filter(
        brush
    ).properties(
            width=width,
            height=height
    )

    bars_generation = alt.Chart(df_cv).mark_bar().encode(
        y='generation',
        color='Predicted soldPrice (Quintile)',
        x='count(Predicted soldPrice (Quintile))'
    ).transform_filter(
        brush
    ).properties(
            width=width,
            height=height
    )

    bars_summons = alt.Chart(df_cv).mark_bar().encode(
        y='summons',
        color='Predicted soldPrice (Quintile)',
        x='count(Predicted soldPrice (Quintile))'
    ).transform_filter(
        brush
    ).properties(
            width=width,
            height=height
    )

    bars_buyhour = alt.Chart(df_cv).mark_bar().encode(
        y='buyHour',
        color='Predicted soldPrice (Quintile)',
        x='count(Predicted soldPrice (Quintile))'
    ).transform_filter(
        brush
    ).properties(
            width=width,
            height=height
    )
    
    return (
        (points & bars_quantile & bars_profession & bars_rarity & bars_mainclass & bars_generation & bars_summons & bars_buyhour)
        .configure(
            background='#100f21'
        ).configure_axis(
            labelColor='white',
            titleColor='white'
        ).configure_legend(
            strokeColor='gray',
            fillColor='#100f21',
            labelColor='white',
            orient='top',
            padding=30,
            #cornerRadius=10,
        )
    )