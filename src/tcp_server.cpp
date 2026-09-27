#include "tcp_server.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <vector>

TcpServer::TcpServer(const std::string& ip, uint16_t port)
    : ip_(ip),
      port_(port),
      server_fd_(-1),
      client_fd_(-1)
{
}


TcpServer::~TcpServer()
{
    closeClient();
    stop();
}


bool TcpServer::start()
{
    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);

    if (server_fd_ < 0)
    {
	perror("socket");
	return false;
    }

    int reuse = 1;

    if (setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0)
    {
	perror("setsockopt");
	return false;
    }

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port_);

    if (ip_ == "0.0.0.0")
    {
	server_addr.sin_addr.s_addr = INADDR_ANY;
    }
    else
    {
	if (inet_pton(AF_INET, ip_.c_str(), &server_addr.sin_addr) <= 0)
	{
	    std::cerr<<"Invalid IP address:"<<ip_<<std::endl;
	    return false;
	}
    }

    if (bind(server_fd_, reinterpret_cast<sockaddr*>(&server_addr), sizeof(server_addr)) < 0)
    {
	perror("bind");
	return false;
    }

    if (listen(server_fd_, 1) < 0)
    {
	perror("listen");
	return false;
    }

    std::cout << "[TCP] Server listening on "
              << ip_ << ":" << port_ << std::endl;

    return true;
}


bool TcpServer::acceptClient()
{
    sockaddr_in client_addr{};
    socklen_t client_len = sizeof(client_addr);

    client_fd_ = accept(server_fd_, reinterpret_cast<sockaddr*>(&client_addr), &client_len);

    if (client_fd_ < 0)
    {
	perror("accept");
	return false;
    }

    char client_ip[INET_ADDRSTRLEN]{};

    inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, sizeof(client_ip));

    std::cout << "[TCP] Client connected: "
              << client_ip << ":"
              << ntohs(client_addr.sin_port)
              << std::endl;

    return true;
}


bool TcpServer::receiveAll(uint8_t* data, size_t size)
{
    size_t received = 0;

    while (received < size)
    {
	ssize_t ret = recv(client_fd_, data + received, size - received, 0);

	if (ret <= 0)
	{
	    return false;
	}

	received += static_cast<size_t>(ret);
    }

    return true;
}


bool TcpServer::sendAll(const uint8_t* data, size_t size)
{
    size_t sent = 0;

    while (sent < size)
    {
	ssize_t ret = send(client_fd_, data + sent, size - sent, 0);

        if (ret <= 0)
        {
            return false;
        }

	sent += static_cast<size_t>(ret);
    }

    return true;
}


bool TcpServer::receiveMessage(std::vector<uint8_t>& data)
{
    uint32_t network_size = 0;

    if (!receiveAll(reinterpret_cast<uint8_t*>(&network_size), sizeof(network_size)))
    {
	return false;
    }

    uint32_t payload_size = ntohl(network_size);

    if (payload_size == 0 || payload_size > 50 * 1024 * 1024)
    {
        std::cerr << "[TCP] Invalid payload size: "
                  << payload_size << std::endl;

        return false;
    }

    data.resize(payload_size);

    return receiveAll(data.data(), data.size());
}


bool TcpServer::sendMessage(const std::vector<uint8_t>& data)
{
    uint32_t payload_size =
        static_cast<uint32_t>(data.size());

    uint32_t network_size = htonl(payload_size);

    if (!sendAll(
            reinterpret_cast<const uint8_t*>(&network_size),
            sizeof(network_size)))
    {
        return false;
    }

    if (data.empty())
    {
        return true;
    }

    return sendAll(
        data.data(),
        data.size()
    );
}


void TcpServer::closeClient()
{
    if (client_fd_ >= 0)
    {
        close(client_fd_);
        client_fd_ = -1;
    }
}


void TcpServer::stop()
{
    if (server_fd_ >= 0)
    {
        close(server_fd_);
        server_fd_ = -1;
    }
}
