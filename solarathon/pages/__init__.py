import solara
from solara.alias import rv

@solara.component
def SharedComponent():
    with solara.Card(solara.Text(text="Welcome, the future awaits"), style={"max-width": "500px"}, margin=0, classes=["my-2"]):
        url = "https://github.com/FelipeCabelloE/solarathon-team5/assets/98831429/dcd12b29-165c-4933-8721-ebffa15e8ba7"
        rv.Img(src=url, contain=True, max_height="400px")
        solara.Markdown(
            f'''
        Videos obtained from:
        
        - [GolfDB](https://github.com/wmcnally/GolfDB)

        - Mikel D. Rodriguez, Javed Ahmed, and Mubarak Shah, Action MACH: A Spatio-temporal Maximum Average Correlation Height Filter for Action Recognition, Computer Vision and Pattern Recognition, 2008.
        
        - Khurram Soomro and Amir R. Zamir, Action Recognition in Realistic Sports Videos, Computer Vision in Sports. Springer International Publishing, 2014.
        ''')

# in case you want to override the default order of the tabs
route_order = ["/", 'video_dashboard']

@solara.component
def Page():
    with solara.Sidebar():
        solara.Text("")

    with solara.Column(style={"padding-top": "30px"}):
        solara.Title("Quiet Heart (Solara(thon) Team 5)")

        with solara.Link("video_dashboard"):
            solara.Button(label='Video dashboard',)
            
        SharedComponent()


@solara.component
def Layout(children):
    # this is the default layout, but you can override it here, for instance some extra padding
    return solara.AppLayout(children=children, style={"padding": "30px"}, sidebar_open=False)
