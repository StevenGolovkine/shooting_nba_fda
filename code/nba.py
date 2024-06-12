#!/usr/bin/env python
# -*-coding:utf8 -*
"""
Functions to scrap nba.com
--------------------------

"""

# Load packages
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from FDApy.representation.functional_data import DenseFunctionalData, BasisFunctionalData
from FDApy.preprocessing import MFPCA

class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        # Note also that we must extrapolate beyond vmin/vmax
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1.]
        return np.ma.masked_array(
            np.interp(value, x, y, left=-np.inf, right=np.inf)
        )

    def inverse(self, value):
        y, x = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.interp(value, x, y, left=-np.inf, right=np.inf)

class NbaScraper:
    """ Class to scrape data from the NBA official website.
    """
    @staticmethod
    def get_json_from_name(name: str, is_player=True) -> int:
        """ Get the json of a player or team from his name
        """
        from nba_api.stats.static import players, teams
        if is_player:
            nba_players = players.get_players()
            return [
                player for player in nba_players
                if player['full_name'] == name
            ][0]
        else:
            nba_teams = teams.get_teams()
            return [
                team for team in nba_teams if team['full_name'] == name
            ][0]
    
    @staticmethod
    def get_player_career(player_id: int) -> list:
        """ Get the career of a player from his id
        """
        from nba_api.stats.endpoints import playercareerstats
        career = playercareerstats.PlayerCareerStats(player_id=player_id)
        return career.get_data_frames()[0]
    
    @staticmethod
    def get_shot_data(id: int, team_ids: list, seasons: list) -> list:
        """ Get the shot data of a player from his id and seasons
        """
        from nba_api.stats.endpoints import shotchartdetail
        df = pd.DataFrame()
        for season in seasons:
            for team in team_ids:
                shot_data = shotchartdetail.ShotChartDetail(
                    team_id=team,
                    player_id=id,
                    context_measure_simple='FGA',
                    season_nullable=season
                )
                df = pd.concat([df, shot_data.get_data_frames()[0]])
        
        return df
    
    @staticmethod
    def get_all_ids(only_active=True) -> list:
        """ Get all the ids of the players
        """
        from nba_api.stats.static import players
        nba_players = players.get_players()
        if only_active:
            return [
                player['id'] for player in nba_players if player['is_active']
            ]
        return [player['id'] for player in nba_players]
    
    @staticmethod
    def get_player_headshot(id: int) -> str:
            """ Get the headshot of a player from his id
            """
            from nba_api.stats.static import players
            import requests
            import shutil
            
            url = f'https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{id}.png'
            output_path = f'../data/nba/transient/headshots/{id}.png'
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                with open(output_path, 'wb') as f:
                    r.raw.decode_content = True
                    shutil.copyfileobj(r.raw, f)
    
    @staticmethod                                    
    def get_all_nba_headshots(only_active=False) -> None:
        """ Get the headshots of all the players
        """
        ids = NbaScraper.get_all_ids(only_active=only_active)
        for id in ids:
            NbaScraper.get_player_headshot(id)


class ShotCharts:
    def __init__(self) -> None:
        pass

    def create_court(ax: mpl.axes, color="white") -> mpl.axes:
        """ Create a basketball court in a matplotlib axes
        """
        # Short corner 3PT lines
        ax.plot([-220, -220], [0, 140], linewidth=2, color=color)
        ax.plot([220, 220], [0, 140], linewidth=2, color=color)
        # 3PT Arc
        ax.add_artist(
            mpl.patches.Arc(
                (0, 140), 440, 315,
                theta1=0, theta2=180,
                facecolor='none', edgecolor=color,
                lw=2
            )
        )
        # Lane and Key
        ax.plot([-80, -80], [0, 190], linewidth=2, color=color)
        ax.plot([80, 80], [0, 190], linewidth=2, color=color)
        ax.plot([-60, -60], [0, 190], linewidth=2, color=color)
        ax.plot([60, 60], [0, 190], linewidth=2, color=color)
        ax.plot([-80, 80], [190, 190], linewidth=2, color=color)
        ax.plot([-250, 250], [0, 0], linewidth=4, color='white')
        ax.add_artist(
            mpl.patches.Circle(
                (0, 190), 60, facecolor='none', edgecolor=color, lw=2
            )
        )
        # Rim
        ax.add_artist(
            mpl.patches.Circle(
                (0, 60), 15, facecolor='none', edgecolor=color, lw=2
            )
        )
        # Backboard
        ax.plot([-30, 30], [40, 40], linewidth=2, color=color)
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        # Set axis limits
        ax.set_xlim(-250, 250)
        ax.set_ylim(0, 422.5)
        return ax

    def add_headshot(ax: mpl.axes, id: int) -> mpl.axes:
        from PIL import Image 
        from urllib import request
        BASE = "https://cdn.nba.com/headshots/nba/latest/260x190"
        headshot_path = f"{BASE}/{str(id)}.png"
        im = Image.open(request.urlopen(headshot_path))

        ax = ax.inset_axes([0.06, -0.04, 0.3, 0.3])
        ax.imshow(im)
        ax.axis('off')
        return ax

    def shots_chart_points(
        ax: mpl.axes,
        df: pd.DataFrame,
        name: str,
        add_headshot: bool = True
    ) -> mpl.axes:
        cond = df.PLAYER_NAME == name
        X = df[cond]['LOC_X'].to_numpy()
        Y = df[cond]['LOC_Y'].to_numpy()
        MADE = df[cond]['SHOT_MADE_FLAG'].to_numpy()
        idx = df[cond]['PLAYER_ID'].iloc[0]

        ax = ShotCharts.create_court(ax, 'black')
        scatter = ax.scatter(X, Y, c=MADE, alpha=1, s=0.2, cmap='viridis')
        
        ax.text(0.03, 0.925, f"Shots chart", fontsize='large', transform=ax.transAxes)
        #ax.text(0.03, 0.875, f"{name}", fontsize='medium', transform=ax.transAxes)

        new_legend = (scatter.legend_elements()[0], ['Missed', 'Made'])
        leg = ax.legend(*new_legend, loc="upper right", title="")
        ax.add_artist(leg)
        
        if add_headshot:
            ax = ShotCharts.add_headshot(ax, idx)
        return ax

    def shots_chart(
        ax: mpl.axes,
        df: pd.DataFrame,
        name: str,
        title: str,
        add_headshot: bool = True
    ) -> mpl.axes:
        cond = df.PLAYER_NAME == name
        density = df[cond]['DENSITY'].iloc[0]
        idx = df[cond]['PLAYER_ID'].iloc[0]

        X_MIN, X_MAX = (-250, 250) 
        Y_MIN, Y_MAX = (0, 422.5)
        XX, YY = np.mgrid[X_MIN:X_MAX:201j, Y_MIN:Y_MAX:201j]

        midnorm = MidpointNormalize(vmin=-1., vcenter=0, vmax=1)
        ax = ShotCharts.create_court(ax, 'black')
        ax.contourf(
            XX, YY, density / np.max(np.abs(density)),
            levels=30,
            cmap='seismic', norm=midnorm
        )
        ax.text(0.03, 0.925, f"{title}", fontsize='large', transform=ax.transAxes)
        #ax.text(0.03, 0.875, f"{name}", fontsize='medium', transform=ax.transAxes)
        if add_headshot:
            ax = ShotCharts.add_headshot(ax, idx)
        return ax

    def shots_chart_reconstruction(
        ax: mpl.axes,
        df: pd.DataFrame,
        df_reconstruction: DenseFunctionalData,
        name: str,
        title: str,
        add_headshot: bool = True
    ) -> mpl.axes:
        cond = df.PLAYER_NAME == name
        idx = df[cond]['PLAYER_ID'].iloc[0]
        indice = df.index[cond]
        density = df_reconstruction[indice[0]].values.squeeze()

        X_MIN, X_MAX = (-250, 250) 
        Y_MIN, Y_MAX = (0, 422.5)
        XX, YY = np.mgrid[X_MIN:X_MAX:201j, Y_MIN:Y_MAX:201j]

        midnorm = MidpointNormalize(vmin=-1., vcenter=0, vmax=1)
        ax = ShotCharts.create_court(ax, 'black')
        ax.contourf(
            XX, YY, density / np.max(np.abs(density)),
            levels=30,
            cmap='seismic', norm=midnorm
        )
        ax.text(0.03, 0.925, f"{title}", fontsize='large', transform=ax.transAxes)
        ax.text(0.03, 0.875, f"{name}", fontsize='medium', transform=ax.transAxes)
        if add_headshot:
            ax = ShotCharts.add_headshot(ax, idx)
        return ax

    def mean_chart(
        ax: mpl.axes,
        mfpca: MFPCA,
        idx_c: int,
        title: str
    ) -> mpl.axes:
        mean = mfpca.mean.data[idx_c][0].values.squeeze()

        X_MIN, X_MAX = (-250, 250) 
        Y_MIN, Y_MAX = (0, 422.5)
        XX, YY = np.mgrid[X_MIN:X_MAX:201j, Y_MIN:Y_MAX:201j]

        midnorm = MidpointNormalize(vmin=-1., vcenter=0, vmax=1)
        ax = ShotCharts.create_court(ax, 'black')
        ax.contourf(
            XX, YY, mean / np.max(np.abs(mean)),
            levels=30,
            cmap='seismic', norm=midnorm
        )
        ax.text(
            0.03, 0.925, f"{title}", fontsize='large', transform=ax.transAxes
        )
        ax.text(
            0.03, 0.87,
            f"$\mu^{{({{{idx_c + 1}}})}}$",
            fontsize='medium', transform=ax.transAxes
        )
        return ax

    def components_chart(
        ax: mpl.axes,
        mfpca: MFPCA,
        idx: int,
        idx_c: int,
        title: str
    ) -> mpl.axes:
        if isinstance(mfpca.eigenfunctions.data[idx_c], BasisFunctionalData):
            eigenfunctions = mfpca.eigenfunctions.data[idx_c].to_grid()
        else:
            eigenfunctions = mfpca.eigenfunctions.data[idx_c]
        eigenfunctions = eigenfunctions[idx].values.squeeze()
        pct = 100 * mfpca.eigenvalues / np.sum(mfpca.eigenvalues)

        X_MIN, X_MAX = (-250, 250) 
        Y_MIN, Y_MAX = (0, 422.5)
        XX, YY = np.mgrid[X_MIN:X_MAX:201j, Y_MIN:Y_MAX:201j]

        midnorm = MidpointNormalize(vmin=-1., vcenter=0, vmax=1)
        ax = ShotCharts.create_court(ax, 'black')
        ax.contourf(
            XX, YY, eigenfunctions / np.max(np.abs(eigenfunctions)),
            levels=30,
            cmap='seismic', norm=midnorm
        )
        ax.text(
            0.03, 0.925, f"{title}", fontsize='large', transform=ax.transAxes
        )
        ax.text(
            0.03, 0.87,
            f"$\phi_{{{idx + 1}}}^{{({{{idx_c + 1}}})}} ({pct[idx]:.1f}\%)$",
            fontsize='medium', transform=ax.transAxes
        )
        return ax

    def shots_decomposition_chart(
        ax: mpl.axes,
        mfpca: MFPCA,
        scores: np.ndarray,
        idx: int,
        idx_c: int,
        name: str,
        title: str,
        maximum: float
    ) -> plt.figure:
        if isinstance(mfpca.eigenfunctions.data[idx_c], BasisFunctionalData):
            eigenfunctions = mfpca.eigenfunctions.data[idx_c].to_grid()
        else:
            eigenfunctions = mfpca.eigenfunctions.data[idx_c]
        eigenfunctions = eigenfunctions[idx].values.squeeze()
        pct = 100 * mfpca.eigenvalues / np.sum(mfpca.eigenvalues)
        score = scores.loc[scores.PLAYER_NAME == name, idx].values
        fpc = score * eigenfunctions
        
        X_MIN, X_MAX = (-250, 250) 
        Y_MIN, Y_MAX = (0, 422.5)
        XX, YY = np.mgrid[X_MIN:X_MAX:201j, Y_MIN:Y_MAX:201j]
        
        midnorm = MidpointNormalize(vmin=-1., vcenter=0, vmax=1)
        ax = ShotCharts.create_court(ax, 'black')
        ax.contourf(
            XX, YY, fpc / maximum,
            levels=30,
            cmap='seismic', norm=midnorm,
        )
        ax.text(0.03, 0.925, f"{title}", fontsize='large', transform=ax.transAxes)
        ax.text(
            0.03, 0.87,
            f"$\mathfrak{{c}}_{{{idx + 1}}}\phi_{{{idx + 1}}}^{{({{{idx_c + 1}}})}} ({pct[idx]:.1f}\%)$",
            fontsize='medium', transform=ax.transAxes
        )
        return ax
