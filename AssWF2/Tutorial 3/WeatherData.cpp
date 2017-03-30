#include "WeatherData.h"

WeatherData::WeatherData()
{
}


WeatherData::~WeatherData()
{
}


string WeatherData::getWeatherStation() {
	return weatherStation;
}

int WeatherData::getYearCollected() {
	return yearCollected;
}

int WeatherData::getMonth() {
	return month;
}

int WeatherData::getDay() {
	return day;
}

int WeatherData::getTime() {
	return time;
}

float WeatherData::getTemp() {
	return temp;
}


void WeatherData::setWeatherStation(string s) {
	weatherStation = s;
}

void WeatherData::setYearCollected(int i) {
	yearCollected = i;
}

void WeatherData::setMonth(int i) {
	month = i;
}

void WeatherData::setDay(int i) {
	day = i;
}

void WeatherData::setTime(int i) {
	time = i;
}

void WeatherData::setTemp(float f) {
	temp = f;
}

