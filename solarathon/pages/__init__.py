import solara



@solara.component
def SharedComponent():
    with solara.Card("Shown on each page", style={"max-width": "500px"}, margin=0, classes=["my-2"]):
        solara.Markdown(
            f'''
            
        
        Videos obtained from:
        
        - [GolfDB](https://github.com/wmcnally/GolfDB)

        - Mikel D. Rodriguez, Javed Ahmed, and Mubarak Shah, Action MACH: A Spatio-temporal Maximum Average Correlation Height Filter for Action Recognition, Computer Vision and Pattern Recognition, 2008.
        
        - Khurram Soomro and Amir R. Zamir, Action Recognition in Realistic Sports Videos, Computer Vision in Sports. Springer International Publishing, 2014.
        ''')
        







# Declare reactive variables at the top level. Components using these variables
# will be re-executed when their values change.
sentence = solara.reactive("Solara makes our team more productive.")
word_limit = solara.reactive(10)


# in case you want to override the default order of the tabs
route_order = ["/", 'video_dashboard']

@solara.component
def Page():

    with solara.Column(style={"padding-top": "30px"}):
        solara.Title("Quiet Heart (Solara(thon) Team 5)")

        with solara.Link("video_dashboard"):
            solara.Button(label='Video dashboard',)
            
        SharedComponent()


@solara.component
def Layout(children):
    # this is the default layout, but you can override it here, for instance some extra padding
    return solara.AppLayout(children=children, style={"padding": "20px"})
