#pragma once
#include <string>


using namespace std;
class WeatherData
{
	string weatherStation;
	int yearCollected;
	int month;
	int day;
	int time;
	float temp;
public:
	WeatherData();
	string getWeatherStation();
	int getYearCollected();
	int getMonth();
	int getDay();
	int getTime();
	float getTemp();
	void setWeatherStation(string s);
	void setYearCollected(int i);
	void setMonth(int i);
	void setDay(int i);
	void setTime(int i);
	void setTemp(float f);
	~WeatherData();
};
